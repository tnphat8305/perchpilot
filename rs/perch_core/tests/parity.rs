use perch_core::{
    frames::{ActionScale, Odometry, ThrustModel},
    model::{Belief, C5_ROWS, C5_V2, C5Samples, Policy},
    perception::{Perception, TagPose},
    phase::{Input, Machine, Phase, PhaseConfig},
    q2::{Imu, Q2Config, Rule, Sensors, Verifier},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    env, fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

const CHECK_ENV: &str = "PERCH_RUST_CHECK";
const TOLERANCE: f64 = 1e-4;

fn fixtures() -> PathBuf {
    PathBuf::from(
        env::var_os(CHECK_ENV)
            .expect("set PERCH_RUST_CHECK to the dump_reference.py output directory"),
    )
}

fn reference_batch(root: &Path, name: &str, width: usize) -> Vec<f32> {
    let bytes = fs::read(root.join(format!("{name}.f32"))).unwrap();
    assert_eq!(bytes.len(), 1000 * 4 * width, "{name} fixture size");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "needs the offline checkpoint and SITL fixtures from dump_reference.py"]
fn torch_weights_parity() {
    let root = fixtures();
    for source in ["", "-pt"] {
        let policy_dir = root.join(format!("policy{source}"));
        let belief_dir = root.join(format!("belief{source}"));
        if source == "-pt" && !policy_dir.exists() && !belief_dir.exists() {
            continue;
        }
        let mut policy = Policy::load(&policy_dir).unwrap();
        let mut belief = Belief::load(&belief_dir).unwrap();
        let n = policy.meta.obs_dim;
        for input in ["random", "sitl"] {
            let data = reference_batch(&root, &format!("policy_{input}"), n + 4);
            let mut err = 0.0_f32;
            for row in data.chunks_exact(n + 4) {
                for (a, &b) in policy.infer(&row[..n]).into_iter().zip(&row[n..]) {
                    assert!(a.is_finite());
                    err = err.max((a - b).abs());
                }
            }
            eprintln!("policy{source} {input}: max action error {err:.8e}");
            assert!(f64::from(err) <= TOLERANCE);
            let n_ch = belief.channels;
            let features = n_ch * C5_ROWS;
            let data = reference_batch(&root, &format!("belief_{input}"), features + 3);
            let mut err = [0.0_f32; 3];
            let mut window: C5Samples = [[0.0; C5_V2]; C5_ROWS];
            for row in data.chunks_exact(features + 3) {
                for (sample, x) in window.iter_mut().zip(row[..features].chunks_exact(n_ch)) {
                    sample[..n_ch].copy_from_slice(x);
                }
                let pred = belief.infer(&window);
                for (i, e) in err.iter_mut().enumerate() {
                    assert!(pred[i].is_finite());
                    *e = e.max((pred[i] - row[features + i]).abs());
                }
            }
            eprintln!(
                "belief{source} {input}: max p_hold {:.8e}, p_slip {:.8e}, margin {:.8e}",
                err[0], err[1], err[2]
            );
            assert!(err[..2].iter().all(|&x| f64::from(x) <= TOLERANCE));
        }
    }
}

#[derive(Deserialize)]
struct Case {
    scenario: String,
    config: PhaseConfig,
    phase: Phase,
    q2: Q2Config,
    pad_offset: f32,
    freeze_plate_v: bool,
}

#[derive(Deserialize)]
struct Sample {
    t: f64,
    imu: Imu,
    hold: Vec<Vec<f64>>,
    x: Vec<f64>,
}

#[derive(Deserialize)]
struct Tag {
    age: f64,
    #[serde(flatten)]
    msg: TagPose,
}

#[derive(Deserialize)]
struct Tick {
    t: f64,
    odom: Option<Odometry>,
    tag: Option<Tag>,
    sensors: Vec<Sample>,
    cmd: Option<String>,
    armed: bool,
    offboard: bool,
    rates_mode: bool,
    output: Value,
}

fn compare(a: &Value, b: &Value, context: &str) -> f64 {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            let err = (a.as_f64().unwrap() - b.as_f64().unwrap()).abs();
            assert!(
                err <= TOLERANCE,
                "{context}: expected {b}, got {a} (error {err:.9e})"
            );
            err
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{context}");
            a.iter()
                .zip(b)
                .enumerate()
                .map(|(i, (a, b))| compare(a, b, &format!("{context}[{i}]")))
                .fold(0.0, f64::max)
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len(), "{context}: {a:?} != {b:?}");
            b.iter()
                .map(|(k, v)| compare(&a[k], v, &format!("{context}.{k}")))
                .fold(0.0, f64::max)
        }
        _ => {
            assert_eq!(a, b, "{context}");
            0.0
        }
    }
}

struct Replay {
    case: Case,
    perception: Perception,
    sensors: Sensors,
    machine: Machine,
    count: usize,
    max_error: f64,
    max_command_error: f64,
    phases: BTreeSet<String>,
    decisions: BTreeSet<String>,
}

impl Replay {
    fn new(root: &Path, case: Case) -> Self {
        let policy = Policy::load(&root.join("policy")).unwrap();
        let scale = ActionScale {
            rate_max: policy.meta.rate_max.into(),
            tw_ratio: policy.meta.tw_ratio_nominal,
            hover: 0.6,
            model: ThrustModel::X500Perch,
        };
        let verifier = (case.q2.rule != Rule::Placeholder).then(|| {
            let belief =
                (case.q2.rule == Rule::Dthr).then(|| Belief::load(&root.join("belief")).unwrap());
            Verifier::new(case.q2, belief).unwrap()
        });
        let machine = Machine::new(
            case.config,
            policy,
            scale,
            case.pad_offset,
            case.phase,
            verifier,
        );
        Self {
            case,
            perception: Perception::new(0.1),
            sensors: Sensors::default(),
            machine,
            count: 0,
            max_error: 0.0,
            max_command_error: 0.0,
            phases: BTreeSet::new(),
            decisions: BTreeSet::new(),
        }
    }

    fn step(&mut self, tick: Tick) {
        let Tick {
            t,
            odom,
            tag,
            sensors,
            cmd,
            armed,
            offboard,
            rates_mode,
            output,
        } = tick;
        let label = format!("{} {t:.2}", self.case.scenario);
        match cmd.as_deref() {
            Some("start") => {
                assert!(self.machine.start_trial(t));
                self.perception.reset_plate();
            }
            Some("reset") => self.machine.reset_trial(),
            None => {}
            _ => panic!("unexpected replay command {cmd:?}"),
        }
        for sample in sensors {
            self.sensors.imu(sample.imu);
            for msg in sample.hold {
                self.sensors.hold(&msg, sample.t);
            }
            let x = self
                .sensors
                .sample(sample.t, self.machine.q2_channels())
                .unwrap();
            self.max_error =
                self.max_error
                    .max(compare(&json!(x), &json!(sample.x), &format!("{label} C5")));
        }
        if let Some(msg) = odom {
            assert!(self.perception.odometry(msg, t));
        }
        if let Some(tag) = tag {
            assert!(self.perception.tag(tag.msg, t, tag.age));
        }
        self.perception.freeze_velocity(
            self.case.freeze_plate_v && matches!(self.machine.phase, Phase::Perch | Phase::Verify),
        );
        let drone = self.perception.drone(t);
        let win = if self.machine.phase == Phase::Verify {
            self.sensors.window()
        } else {
            None
        };
        let mut out = self
            .machine
            .step(Input {
                t,
                drone,
                armed,
                offboard,
                plate: self.perception.plate(t),
                contact: self.sensors.contact(t),
                rates_mode: Some(rates_mode),
                window: win.as_ref(),
            })
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        out.apply_position_handoff(self.machine.scale, drone, Some(rates_mode));
        let obs = out.action.map(|_| self.machine.observation());
        let actual = json!({
            "phase": out.phase, "mode": out.mode, "pos": out.pos, "vel": out.vel,
            "yaw": out.yaw, "rates": out.rates, "thrust": out.thrust,
            "thrust_body": out.thrust.map(|x| [0.0, 0.0, -x]),
            "commands": out.commands, "hold_cmd": out.hold_cmd,
            "action": out.action, "obs": obs, "q2": self.machine.q2_result,
            "k": self.machine.k, "switching": self.machine.switching(),
            "ready": self.machine.ready, "result": self.machine.result,
        });
        self.max_error = self.max_error.max(compare(&actual, &output, &label));
        for key in [
            "pos",
            "vel",
            "yaw",
            "rates",
            "thrust",
            "thrust_body",
            "commands",
            "hold_cmd",
        ] {
            self.max_command_error =
                self.max_command_error
                    .max(compare(&actual[key], &output[key], &label));
        }
        self.phases.insert(out.phase.to_string());
        if let Some(res) = &self.machine.q2_result {
            self.decisions.insert(res.decision.as_str().into());
        }
        self.count += 1;
    }

    fn finish(self) -> String {
        assert!(self.count > 0);
        let name = self.case.scenario;
        match name.as_str() {
            "full_cycle" | "position_depart" => {
                for phase in [
                    "WAIT", "APPROACH", "PERCH", "VERIFY", "REST", "DEPART", "DONE",
                ] {
                    assert!(self.phases.contains(phase), "{name} missed {phase}");
                }
            }
            "b4e_abort" => assert!(self.decisions.contains("ABORT")),
            "dthr_off" | "b4b_off" | "none_off" => assert!(self.decisions.contains("OFF")),
            "d12_contact_loss" => assert!(self.decisions.contains("SELF_ABORT")),
            _ => panic!("unexpected replay scenario {name}"),
        }
        eprintln!(
            "replay {name}: {} ticks, exact phase and decisions, max command error {:.8e}, all numeric {:.8e}",
            self.count, self.max_command_error, self.max_error
        );
        name
    }
}

#[test]
#[ignore = "needs the offline replay fixtures from dump_reference.py"]
fn python_node_replay_parity() {
    let root = fixtures();
    let reader = BufReader::new(fs::File::open(root.join("replay.jsonl")).unwrap());
    let mut replay: Option<Replay> = None;
    let mut tested = BTreeSet::new();
    for line in reader.lines() {
        let row: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if row.get("scenario").is_some() {
            if let Some(r) = replay.take() {
                tested.insert(r.finish());
            }
            replay = Some(Replay::new(&root, serde_json::from_value(row).unwrap()));
        } else {
            replay
                .as_mut()
                .unwrap()
                .step(serde_json::from_value(row).unwrap());
        }
    }
    tested.insert(replay.unwrap().finish());
    for name in [
        "full_cycle",
        "b4e_abort",
        "dthr_off",
        "d12_contact_loss",
        "b4b_off",
        "none_off",
        "position_depart",
    ] {
        assert!(tested.contains(name));
    }
}
