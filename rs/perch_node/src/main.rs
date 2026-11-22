mod telemetry;

use clap::Parser;
use futures::{
    StreamExt,
    executor::{LocalPool, LocalSpawner},
    task::LocalSpawnExt,
};
use perch_core::{
    frames::{
        ActionScale, Drone, Mat3, Odometry, ThrustModel, Vec3, ned_enu, rot_to_quat, rot_y,
        yaw_enu_ned,
    },
    model::{Belief, Policy},
    perception::{Perception, TAG_PERIOD, TagPose},
    phase::{Input, Machine, Mode, Output, Phase, PhaseConfig},
    q2::{Imu, Q2Config, Rule, Sensors},
};
use r2r::{
    Clock, ClockType, Context, NativeMsg, Node, Publisher, QosProfile, WrappedTypesupport,
    geometry_msgs::msg::PoseStamped,
    px4_msgs::msg::{
        OffboardControlMode, SensorCombined, TrajectorySetpoint, VehicleCommand,
        VehicleControlMode, VehicleOdometry, VehicleRatesSetpoint, VehicleStatus,
    },
    std_msgs::msg::{Float64, Float64MultiArray, String as RosString},
};
use serde::Deserialize;
use std::{
    cell::RefCell,
    env, io,
    path::{Path, PathBuf},
    process::Command,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use telemetry::{Jsonl, Snapshot, Telemetry};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
const NODE_NAME: &str = "perchpilot_perch_node";

fn parse_enum<T: for<'a> Deserialize<'a>>(s: &str) -> std::result::Result<T, String> {
    T::deserialize(serde::de::value::StrDeserializer::<serde::de::value::Error>::new(s))
        .map_err(|e| e.to_string())
}

#[derive(Parser)]
#[command(about = "PerchPilot ROS 2 flight node")]
struct Args {
    #[arg(long, visible_alias = "policy-run")]
    run: PathBuf,
    #[arg(long, default_value = "auto", value_parser = ["auto", "rust"])]
    backend: String,
    #[arg(long, default_value = "/px4_7")]
    ns: String,
    #[arg(long, default_value_t = 8)]
    target_system: u8,
    #[arg(long, default_value_t = 7)]
    px4_instance: u32,
    #[arg(long)]
    hover_thrust: Option<f32>,
    #[arg(long, default_value = "linear", value_parser = parse_enum::<ThrustModel>)]
    thrust_model: ThrustModel,
    #[arg(long, default_value_t = 0.05)]
    pad_offset: f32,
    #[arg(long, default_value_t = 100.0)]
    rate_hz: f64,
    #[arg(long, default_value_t = 20.0)]
    diag_hz: f64,
    #[arg(long, default_value = "tag", value_parser = ["tag"])]
    perception: String,
    #[arg(long, default_value_t = 0.1)]
    kf_q_acc: f32,
    #[arg(long, default_value = "placeholder", value_parser = parse_enum::<Rule>)]
    verify_rule: Rule,
    #[arg(
        long,
        default_value = "/home/sup/data/perchpilot/runs/q2_belief_v2_small"
    )]
    q2_belief: PathBuf,
    #[arg(long, default_value_os_t = default_decide_json())]
    q2_decide_json: PathBuf,
    #[arg(long, default_value = "D-thr_v3_early_stop")]
    q2_dthr_key: String,
    #[arg(long)]
    q2_tau: Option<f32>,
    #[arg(long)]
    q2_s: Option<f32>,
    #[arg(long)]
    q2_dmin: Option<usize>,
    #[arg(long, num_args = 3)]
    q2_slip_off: Option<Vec<f32>>,
    #[arg(long)]
    q2_log: Option<PathBuf>,
    #[arg(long)]
    depart_pushoff: bool,
    #[arg(long)]
    freeze_plate_v: bool,
    #[arg(long, default_value_t = 1.0)]
    yaw_scale: f32,
    #[arg(long, default_value_t = 0)]
    extra_delay_steps: usize,
    #[arg(long, default_value_t = 4.0)]
    approach_back: f32,
    #[arg(long, default_value_t = 1.0)]
    approach_up: f32,
    #[arg(long, default_value_t = 5.0)]
    rest_s: f64,
    #[arg(long, default_value_t = 3)]
    max_attempts: u32,
    #[arg(long, default_value_t = 1.0)]
    perch_lost_s: f64,
    #[arg(long)]
    eval_mode: bool,
    #[arg(long, num_args = 3, default_values = ["2.5", "0", "2.2"], allow_negative_numbers = true)]
    standby_offset: Vec<f32>,
    #[arg(long)]
    log_dir: Option<PathBuf>,
    #[arg(long)]
    tick_log: Option<PathBuf>,
    #[arg(long, value_name = "NAME=VALUE")]
    px4_param: Vec<String>,
    #[arg(long, default_value_t = 0.0)]
    duration: f64,
    #[arg(long)]
    dry_run: bool,
}

fn default_decide_json() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/q2_decide_v3_v2_small_ens5.json")
}

fn px4_cli(args: &Args, cmd: &[&str]) -> Option<String> {
    let exe = [
        env::var_os("PX4_DIR").map(PathBuf::from),
        Some("/home/sup/data/perchpilot/PX4-perch".into()),
        Some("/home/sup/PX4-Autopilot".into()),
    ]
    .into_iter()
    .flatten()
    .map(|root| root.join("build/px4_sitl_default/bin/px4-param"))
    .find(|path| path.is_file())?;
    let out = Command::new("timeout")
        .args(["-k", "1s", "5s"])
        .arg(exe)
        .args(["--instance", &args.px4_instance.to_string()])
        .args(cmd)
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()
}

fn read_param(args: &Args, name: &str) -> Option<f64> {
    px4_cli(args, &["show", name])?.lines().find_map(|line| {
        let (_, val) = line.split_once(name)?.1.split_once(':')?;
        val.split_whitespace().next()?.parse().ok()
    })
}

fn set_params(args: &Args) -> Result {
    for item in &args.px4_param {
        let (name, value) = item
            .split_once('=')
            .ok_or("--px4-param expects NAME=VALUE")?;
        let target = value.parse::<f64>()?;
        let mut applied = false;
        for _ in 0..10 {
            px4_cli(args, &["set", name, value]);
            if read_param(args, name).is_some_and(|v| (v - target).abs() < 1e-6) {
                applied = true;
                break;
            }
            thread::sleep(Duration::from_secs(1));
        }
        if !applied {
            return Err(format!("PX4 parameter {item} could not be set").into());
        }
        eprintln!("px4 param {item}: ok");
    }
    Ok(())
}

fn machine(a: &Args) -> Result<Machine> {
    let policy = Policy::load(&a.run)?;
    let hover = a
        .hover_thrust
        .or_else(|| {
            (!a.dry_run)
                .then(|| read_param(a, "MPC_THR_HOVER").map(|v| v as f32))
                .flatten()
        })
        .unwrap_or(0.60);
    let scale = ActionScale {
        rate_max: policy.meta.rate_max.into(),
        tw_ratio: policy.meta.tw_ratio_nominal,
        hover,
        model: a.thrust_model,
    };
    let cfg = PhaseConfig {
        approach_back: a.approach_back,
        approach_up: a.approach_up,
        rest_s: a.rest_s,
        max_attempts: a.max_attempts,
        perch_lost_s: if a.dry_run { 1e9 } else { a.perch_lost_s },
        extra_delay_steps: a.extra_delay_steps,
        yaw_cmd_scale: a.yaw_scale,
        depart_pushoff: a.depart_pushoff,
        standby_offset: a
            .eval_mode
            .then(|| a.standby_offset.as_slice().try_into().unwrap()),
        max_steps: if a.dry_run { 1_000_000_000 } else { 600 },
        ..PhaseConfig::default()
    };
    let verifier = if a.verify_rule == Rule::Placeholder {
        None
    } else {
        let mut cfg = Q2Config::load(a.verify_rule, &a.q2_decide_json, &a.q2_dthr_key)?;
        cfg.tau = a.q2_tau.unwrap_or(cfg.tau);
        cfg.s = a.q2_s.unwrap_or(cfg.s);
        cfg.d_min = a.q2_dmin.unwrap_or(cfg.d_min);
        if a.verify_rule == Rule::FsmB4e {
            cfg.slip_off_mm = a
                .q2_slip_off
                .as_deref()
                .ok_or("--verify-rule fsm_b4e needs --q2-slip-off")?
                .try_into()?;
        }
        let belief = (a.verify_rule == Rule::Dthr)
            .then(|| Belief::load(&a.q2_belief))
            .transpose()?;
        eprintln!(
            "Q2 verify rule {}: tau={} s={} e_thr={:.4} slip_off_mm={:?} schedule={:?}",
            cfg.rule.as_str(),
            cfg.tau,
            cfg.s,
            cfg.e_thr,
            cfg.slip_off_mm,
            cfg.schedule()
        );
        Some(perch_core::q2::Verifier::new(cfg, belief)?)
    };
    eprintln!(
        "policy_backend=rust obs_dim={} hover_thrust={} rate_max={:?} tw_ratio={} thrust_model={:?} max_thrust_g={:.3} perception={} dry_run={}",
        policy.meta.obs_dim,
        scale.hover,
        scale.rate_max.as_slice(),
        scale.tw_ratio,
        scale.model,
        scale.u_to_acc(1.0) / perch_core::G,
        a.perception,
        a.dry_run
    );
    Ok(Machine::new(
        cfg,
        policy,
        scale,
        a.pad_offset,
        if a.dry_run { Phase::Perch } else { Phase::Wait },
        verifier,
    ))
}

// Reuse the fixed-layout PX4 messages at the control rate.
struct Tx<T: WrappedTypesupport + 'static> {
    publisher: Publisher<T>,
    msg: NativeMsg<T>,
}

impl<T: WrappedTypesupport + 'static> Tx<T> {
    fn new(node: &mut Node, topic: &str, qos: QosProfile) -> Result<Self> {
        Ok(Self {
            publisher: node.create_publisher(topic, qos)?,
            msg: NativeMsg::new(),
        })
    }

    fn send(&mut self) -> Result {
        self.publisher.publish_native(&mut self.msg)?;
        Ok(())
    }
}

struct Publishers {
    mode: Tx<OffboardControlMode>,
    traj: Tx<TrajectorySetpoint>,
    rate: Tx<VehicleRatesSetpoint>,
    command: Tx<VehicleCommand>,
    hold: Tx<Float64>,
}

impl Publishers {
    fn new(node: &mut Node, ns: &str, qos: QosProfile) -> Result<Self> {
        Ok(Self {
            mode: Tx::new(
                node,
                &format!("{ns}/fmu/in/offboard_control_mode"),
                qos.clone(),
            )?,
            traj: Tx::new(
                node,
                &format!("{ns}/fmu/in/trajectory_setpoint"),
                qos.clone(),
            )?,
            rate: Tx::new(
                node,
                &format!("{ns}/fmu/in/vehicle_rates_setpoint"),
                qos.clone(),
            )?,
            command: Tx::new(node, &format!("{ns}/fmu/in/vehicle_command"), qos)?,
            hold: Tx::new(node, "/perch/hold_cmd", QosProfile::default())?,
        })
    }

    fn hold(&mut self, enabled: bool) -> Result {
        self.hold.msg.data = f64::from(enabled);
        self.hold.send()
    }

    fn send(&mut self, out: &Output, ts: u64, target_system: u8) -> Result {
        self.mode.msg.timestamp = ts;
        self.mode.msg.position = out.mode == Mode::Position;
        self.mode.msg.body_rate = out.mode == Mode::Rates;
        self.mode.send()?;
        if let Some(p) = out.pos {
            let msg = &mut self.traj.msg;
            msg.timestamp = ts;
            msg.position = ned_enu(p).into();
            msg.velocity = out.vel.map_or([f32::NAN; 3], |v| ned_enu(v).into());
            msg.acceleration = [f32::NAN; 3];
            msg.yaw = out.yaw.map_or(f32::NAN, yaw_enu_ned);
            msg.yawspeed = f32::NAN;
            self.traj.send()?;
        }
        if let Some(rates) = out.rates {
            let msg = &mut self.rate.msg;
            msg.timestamp = ts;
            [msg.roll, msg.pitch, msg.yaw] = rates.into();
            msg.thrust_body = [0.0, 0.0, -out.thrust.unwrap()];
            self.rate.send()?;
        }
        for &(command, param1, param2) in &out.commands {
            let msg = &mut self.command.msg;
            msg.timestamp = ts;
            msg.command = command.into();
            msg.param1 = param1;
            msg.param2 = param2;
            msg.target_system = target_system;
            msg.target_component = 1;
            msg.source_system = 1;
            msg.source_component = 1;
            msg.from_external = true;
            self.command.send()?;
            r2r::log_info!(NODE_NAME, "[cmd] {command} p1={param1} p2={param2}");
        }
        Ok(())
    }
}

struct Runtime {
    args: Args,
    pm: Machine,
    perception: Perception,
    sensors: Sensors,
    pubs: Option<Publishers>,
    diag: Publisher<RosString>,
    steady: Clock,
    ros: Arc<Mutex<Clock>>,
    telemetry: Telemetry,
    q2_log: Jsonl,
    n_odom: usize,
    n_meas: usize,
    armed: bool,
    offboard: bool,
    rates_only: Option<bool>,
    hold_enabled: bool,
    trial: u64,
    next_dry_tag: f64,
    failure: Option<Box<dyn std::error::Error>>,
}

impl Runtime {
    fn now(&mut self) -> Result<f64> {
        Ok(self.steady.get_now()?.as_secs_f64())
    }

    fn log_result(&mut self, result: Result) {
        if let Err(err) = result
            && self.failure.is_none()
        {
            self.failure = Some(err);
        }
    }

    fn odometry(&mut self, msg: Odometry) -> Result {
        let now = self.now()?;
        self.n_odom += usize::from(self.perception.odometry(msg, now));
        Ok(())
    }

    fn tag(&mut self, msg: PoseStamped) -> Result {
        if msg.header.frame_id != "camera" {
            return Ok(());
        }
        let now = self.now()?;
        let stamp = msg.header.stamp;
        let stamp_ns = i64::from(stamp.sec) * 1_000_000_000 + i64::from(stamp.nanosec);
        let age = (ros_ns(&self.ros)? as i64 - stamp_ns) as f64 * 1e-9;
        let (p, q) = (msg.pose.position, msg.pose.orientation);
        let tag = TagPose {
            position: [p.x as f32, p.y as f32, p.z as f32],
            q: [q.w as f32, q.x as f32, q.y as f32, q.z as f32],
        };
        self.n_meas += usize::from(self.perception.tag(tag, now, age));
        Ok(())
    }

    fn on_cmd(&mut self, msg: RosString) -> Result {
        if msg.data == "reset" {
            self.pm.reset_trial();
            return Ok(());
        }
        let mut fields = msg.data.split(':');
        if fields.next() != Some("start") {
            return Ok(());
        }
        let trial = match fields.next() {
            None => self.trial + 1,
            Some(s) => match s.parse() {
                Ok(id) => id,
                Err(_) => return Ok(()),
            },
        };
        let now = self.now()?;
        if self.pm.start_trial(now) {
            self.trial = trial;
            self.perception.reset_plate();
            self.hold_enabled = true;
            self.pubs.as_mut().unwrap().hold(true)?;
        }
        Ok(())
    }

    fn sample(&mut self) -> Result {
        let now = self.now()?;
        if let Some(row) = self.sensors.sample(now, self.pm.q2_channels())
            && matches!(
                self.pm.phase,
                Phase::Perch | Phase::Verify | Phase::Rest | Phase::Depart
            )
        {
            self.q2_log.sample(now, self.trial, self.pm.phase, row)?;
        }
        Ok(())
    }

    fn dry_scene(&mut self, now: f64) -> Drone {
        let r = rot_y(-1.1345);
        let n = r.column(2).into_owned();
        let c = Vec3::new(7.2, 0.0, 1.3) + 0.01 * n;
        let p = c + 2.0 * n;
        if now >= self.next_dry_tag {
            let cam = rot_y(15_f32.to_radians());
            self.perception.odometry(
                Odometry {
                    position: ned_enu(p).into(),
                    q: rot_to_quat(perch_core::frames::ned_frd_enu_flu(Mat3::identity())),
                    velocity: [0.0; 3],
                    angular_velocity: [0.0; 3],
                    pose_frame: 1,
                    velocity_frame: 1,
                },
                now,
            );
            self.n_meas += usize::from(self.perception.tag(
                TagPose {
                    position: (cam.transpose() * (c - p)).into(),
                    q: rot_to_quat(cam.transpose() * r),
                },
                now,
                0.0,
            ));
            self.next_dry_tag = (self.next_dry_tag + TAG_PERIOD).max(now);
            if self.n_meas == 1 {
                self.next_dry_tag = now + TAG_PERIOD;
            }
        }
        Drone {
            p,
            v: Vec3::zeros(),
            r: Mat3::identity(),
            w: Vec3::zeros(),
            t: now,
        }
    }

    fn tick(&mut self) -> Result {
        let start = Instant::now();
        let now = self.now()?;
        let drone = if self.args.dry_run {
            Some(self.dry_scene(now))
        } else {
            self.perception.drone(now)
        };
        self.perception.freeze_velocity(
            self.args.freeze_plate_v && matches!(self.pm.phase, Phase::Perch | Phase::Verify),
        );
        let plate = self.perception.plate(now);
        let window = if self.pm.phase == Phase::Verify {
            self.sensors.window()
        } else {
            None
        };
        let mut out = self.pm.step(Input {
            t: now,
            drone,
            plate,
            armed: self.armed,
            offboard: self.offboard,
            contact: self.sensors.contact(now),
            rates_mode: if self.args.dry_run {
                None
            } else {
                Some(self.rates_only.unwrap_or(false))
            },
            window: window.as_ref(),
        })?;
        out.apply_position_handoff(self.pm.scale, drone, self.rates_only);
        for e in &out.events {
            r2r::log_info!(NODE_NAME, "[phase] {e}");
        }
        if let Some(pubs) = &mut self.pubs
            && out.mode != Mode::None
        {
            let ts = ros_ns(&self.ros)? / 1000;
            pubs.send(&out, ts, self.args.target_system)?;
            if let Some(hold) = out.hold_cmd {
                self.hold_enabled = hold != 0.0;
                pubs.hold(self.hold_enabled)?;
            }
        }

        if let Some(msg) = self.telemetry.record(
            now,
            Snapshot {
                pm: &self.pm,
                out: &out,
                drone,
                plate,
                rates_only: self.rates_only,
                trial: self.trial,
                armed: self.armed,
                offboard: self.offboard,
                window: window.as_ref(),
            },
        )? {
            self.diag.publish(&RosString { data: msg })?;
            if self.args.eval_mode
                && let Some(pubs) = &mut self.pubs
            {
                pubs.hold(self.hold_enabled)?;
            }
        }
        self.telemetry.tick_time(start.elapsed());
        self.telemetry
            .report(now, &self.pm, self.n_odom, self.n_meas)?;
        Ok(())
    }
}

fn subscribe<T, F>(
    node: &mut Node,
    pool: &LocalSpawner,
    rt: &Rc<RefCell<Runtime>>,
    topic: &str,
    qos: QosProfile,
    mut f: F,
) -> Result
where
    T: WrappedTypesupport + 'static,
    F: FnMut(&mut Runtime, T) -> Result + 'static,
{
    let stream = node.subscribe::<T>(topic, qos)?;
    let rt = Rc::clone(rt);
    pool.spawn_local(stream.for_each(move |msg| {
        let mut state = rt.borrow_mut();
        if state.failure.is_none() {
            let result = f(&mut state, msg);
            state.log_result(result);
        }
        std::future::ready(())
    }))?;
    Ok(())
}

fn ros_ns(clock: &Mutex<Clock>) -> Result<u64> {
    Ok(clock.lock().unwrap().get_now()?.as_nanos() as u64)
}

fn subscriptions(
    node: &mut Node,
    pool: &LocalSpawner,
    rt: &Rc<RefCell<Runtime>>,
    qos: QosProfile,
) -> Result {
    let (ns, eval, q2) = {
        let a = &rt.borrow().args;
        (
            a.ns.clone(),
            a.eval_mode,
            a.verify_rule != Rule::Placeholder,
        )
    };
    subscribe(
        node,
        pool,
        rt,
        &format!("{ns}/fmu/out/vehicle_odometry"),
        qos.clone(),
        |s, msg: VehicleOdometry| {
            s.odometry(Odometry {
                position: msg
                    .position
                    .try_into()
                    .map_err(|_| "invalid px4 position array")?,
                q: msg
                    .q
                    .try_into()
                    .map_err(|_| "invalid px4 quaternion array")?,
                velocity: msg
                    .velocity
                    .try_into()
                    .map_err(|_| "invalid px4 velocity array")?,
                angular_velocity: msg
                    .angular_velocity
                    .try_into()
                    .map_err(|_| "invalid px4 angular velocity array")?,
                pose_frame: msg.pose_frame,
                velocity_frame: msg.velocity_frame,
            })
        },
    )?;
    subscribe(
        node,
        pool,
        rt,
        &format!("{ns}/fmu/out/vehicle_status"),
        qos.clone(),
        |s, msg: VehicleStatus| {
            s.armed = msg.arming_state == VehicleStatus::ARMING_STATE_ARMED as u8;
            s.offboard = msg.nav_state == VehicleStatus::NAVIGATION_STATE_OFFBOARD as u8;
            Ok(())
        },
    )?;
    subscribe(
        node,
        pool,
        rt,
        &format!("{ns}/fmu/out/vehicle_control_mode"),
        qos.clone(),
        |s, msg: VehicleControlMode| {
            s.rates_only = Some(
                msg.flag_control_rates_enabled
                    && !msg.flag_control_attitude_enabled
                    && !msg.flag_control_position_enabled,
            );
            Ok(())
        },
    )?;
    if q2 {
        subscribe(
            node,
            pool,
            rt,
            &format!("{ns}/fmu/out/sensor_combined"),
            qos,
            |s, msg: SensorCombined| {
                s.sensors.imu(Imu {
                    acc: msg
                        .accelerometer_m_s2
                        .try_into()
                        .map_err(|_| "invalid px4 IMU acceleration array")?,
                    gyro: msg
                        .gyro_rad
                        .try_into()
                        .map_err(|_| "invalid px4 IMU gyro array")?,
                });
                Ok(())
            },
        )?;
    }
    subscribe(
        node,
        pool,
        rt,
        "/perch/tag_pose",
        QosProfile::default().best_effort().keep_last(1),
        Runtime::tag,
    )?;
    if eval {
        subscribe(
            node,
            pool,
            rt,
            "/perchpilot/cmd",
            QosProfile::default(),
            Runtime::on_cmd,
        )?;
        subscribe(
            node,
            pool,
            rt,
            "/perch/hold_sensors",
            QosProfile::default().reliable().keep_last(50),
            |s, msg: Float64MultiArray| {
                let t = s.now()?;
                s.sensors.hold(&msg.data, t);
                Ok(())
            },
        )?;
    }
    Ok(())
}

fn timer(
    node: &mut Node,
    pool: &LocalSpawner,
    rt: &Rc<RefCell<Runtime>>,
    period: Duration,
    callback: fn(&mut Runtime) -> Result,
) -> Result {
    let mut timer = node.create_wall_timer(period)?;
    let rt = Rc::clone(rt);
    pool.spawn_local(async move {
        while let Ok(_elapsed) = timer.tick().await {
            let mut s = rt.borrow_mut();
            if s.failure.is_none() {
                let result = callback(&mut s);
                s.log_result(result);
            }
        }
    })?;
    Ok(())
}

fn main() -> Result {
    let args = Args::parse();
    if [args.rate_hz, args.diag_hz]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err("--rate-hz and --diag-hz must be positive finite values".into());
    }
    let period = Duration::try_from_secs_f64(1.0 / args.rate_hz)?;
    if period.is_zero() {
        return Err("--rate-hz exceeds timer resolution".into());
    }
    let pm = machine(&args)?;
    set_params(&args)?;
    let ctx = Context::create()?;
    let mut node = Node::create(ctx.clone(), NODE_NAME, "/")?;
    let qos = QosProfile::default()
        .best_effort()
        .transient_local()
        .keep_last(1);
    let diag = node.create_publisher("/perchpilot/diag", QosProfile::default())?;
    let pubs = (!args.dry_run)
        .then(|| Publishers::new(&mut node, &args.ns, qos.clone()))
        .transpose()?;
    let mut steady = Clock::create(ClockType::SteadyTime)?;
    let time = steady.get_now()?.as_secs_f64();
    let rt = Rc::new(RefCell::new(Runtime {
        telemetry: Telemetry::new(
            time,
            args.rate_hz,
            args.diag_hz,
            args.tick_log.clone(),
            args.log_dir.clone(),
            pm.observation().len(),
        )?,
        q2_log: Jsonl::append(args.q2_log.clone()),
        perception: Perception::new(args.kf_q_acc),
        armed: args.dry_run,
        offboard: args.dry_run,
        args,
        pm,
        sensors: Sensors::default(),
        pubs,
        diag,
        steady,
        ros: node.get_ros_clock(),
        n_odom: 0,
        n_meas: 0,
        rates_only: None,
        hold_enabled: false,
        trial: 0,
        next_dry_tag: time,
        failure: None,
    }));
    let mut pool = LocalPool::new();
    let spawner = pool.spawner();
    if !rt.borrow().args.dry_run {
        subscriptions(&mut node, &spawner, &rt, qos)?;
    }
    if rt.borrow().args.verify_rule != Rule::Placeholder {
        timer(
            &mut node,
            &spawner,
            &rt,
            Duration::from_millis(5),
            Runtime::sample,
        )?;
    }
    timer(&mut node, &spawner, &rt, period, Runtime::tick)?;
    let stop = Arc::new(AtomicBool::new(false));
    let sig = Arc::clone(&stop);
    ctrlc::set_handler(move || sig.store(true, Ordering::Relaxed))?;
    let duration = rt.borrow().args.duration;
    let start = Instant::now();
    while !stop.load(Ordering::Relaxed)
        && ctx.is_valid()
        && (duration <= 0.0 || start.elapsed().as_secs_f64() < duration)
    {
        node.spin_once(Duration::from_millis(50));
        pool.run_until_stalled();
        if let Some(err) = rt.borrow_mut().failure.take() {
            return Err(err);
        }
    }
    let mut r = rt.borrow_mut();
    let now = r.now()?;
    let Runtime {
        telemetry,
        pm,
        n_meas,
        ..
    } = &mut *r;
    telemetry.finish(now, pm, *n_meas)?;
    io::Write::flush(&mut io::stdout())?;
    Ok(())
}
