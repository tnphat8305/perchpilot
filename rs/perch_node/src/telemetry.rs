use perch_core::{
    frames::{Drone, Plate, Vec3, rot_to_quat},
    phase::{Machine, Output, Phase},
    q2::{DecisionPoint, Q2Result, Rule, Window, rounded},
};
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

pub struct Jsonl {
    path: Option<PathBuf>,
    file: Option<BufWriter<File>>,
}

impl Jsonl {
    pub fn append(path: Option<PathBuf>) -> Self {
        Self { path, file: None }
    }

    fn create(path: Option<PathBuf>) -> io::Result<Self> {
        let file = path
            .as_ref()
            .map(|p| File::create(p).map(BufWriter::new))
            .transpose()?;
        Ok(Self { path, file })
    }

    fn write(&mut self, row: &impl Serialize) -> io::Result<()> {
        if let Some(path) = &self.path {
            if self.file.is_none() {
                if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                    fs::create_dir_all(parent)?;
                }
                self.file = Some(BufWriter::new(
                    OpenOptions::new().append(true).create(true).open(path)?,
                ));
            }
            let file = self.file.as_mut().unwrap();
            serde_json::to_writer(&mut *file, row)?;
            writeln!(file)?;
            file.flush()?;
        }
        Ok(())
    }

    pub fn sample(&mut self, t: f64, trial: u64, phase: Phase, sample: &[f32]) -> io::Result<()> {
        #[derive(Serialize)]
        struct Sample<'a> {
            wall: f64,
            trial: u64,
            phase: Phase,
            x: &'a [f64],
        }
        if self.path.is_some() {
            let mut x = [0.0; 13];
            for (dst, val) in x.iter_mut().zip(sample) {
                *dst = rounded((*val).into(), 1e5);
            }
            self.write(&Sample {
                wall: rounded(t, 1e5),
                trial,
                phase,
                x: &x[..sample.len()],
            })?;
        }
        Ok(())
    }
}

fn round_array<const N: usize>(v: [f32; N], scale: f64) -> [f64; N] {
    v.map(|x| rounded(x.into(), scale))
}

fn round_vec(v: Vec3, scale: f64) -> [f64; 3] {
    round_array(v.into(), scale)
}

const STATS_WINDOW: usize = 5000;

struct Stats {
    data: Vec<f64>,
    n: usize,
}

impl Stats {
    fn new() -> Self {
        Self {
            data: Vec::with_capacity(STATS_WINDOW),
            n: 0,
        }
    }

    fn add(&mut self, x: f64) {
        if self.data.len() < STATS_WINDOW {
            self.data.push(x);
        } else {
            self.data[self.n % STATS_WINDOW] = x;
        }
        self.n += 1;
    }

    fn summary(&self, scratch: &mut Vec<f64>) -> StatSummary {
        scratch.clear();
        scratch.extend_from_slice(&self.data);
        scratch.sort_unstable_by(f64::total_cmp);
        let quantile = |q: f64| {
            scratch.last().map(|_| {
                let index = q * (scratch.len() - 1) as f64;
                let lo = scratch[index.floor() as usize];
                let hi = scratch[index.ceil() as usize];
                rounded(1e3 * (lo + index.fract() * (hi - lo)), 1e4)
            })
        };
        StatSummary {
            n: self.n,
            p50: quantile(0.5),
            p95: quantile(0.95),
            p99: quantile(0.99),
            max: quantile(1.0),
        }
    }
}

#[derive(Serialize)]
struct StatSummary {
    n: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    p50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p95: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p99: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<f64>,
}

#[derive(Serialize)]
struct Timings {
    loop_ms: StatSummary,
    jitter_ms: StatSummary,
    infer_ms: StatSummary,
    tick_compute_ms: StatSummary,
}

#[derive(Serialize)]
struct Probe {
    #[serde(rename = "macro")]
    macro_name: &'static str,
    last_point: Option<DecisionPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slip_mm: Option<f32>,
}

#[derive(Serialize)]
struct Diag<'a> {
    t: f64,
    phase: Phase,
    attempt: u32,
    rho: f64,
    plate_age: Option<f64>,
    plate_vis: f64,
    tag_visible: bool,
    action: Option<[f64; 4]>,
    thrust_norm: Option<f64>,
    rates_frd: Option<[f64; 3]>,
    loop_dt_ms: Option<f64>,
    infer_ms: Option<f64>,
    armed: bool,
    offboard: bool,
    events: &'a [&'a str],
    trial: u64,
    ready: bool,
    result: Option<&'static str>,
    q2: Option<&'a Q2Result>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p_enu: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plate_c_enu: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plate_v_enu: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    q2_probe: Option<Probe>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b4e_thresholds: Option<B4eThresholds>,
    #[serde(skip_serializing_if = "Option::is_none")]
    loop_ms: Option<StatSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    jitter_ms: Option<StatSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    infer_ms_stats: Option<StatSummary>,
}

#[derive(Serialize)]
struct B4eThresholds {
    #[serde(rename = "RELAX")]
    relax: f32,
    #[serde(rename = "PULL0.3")]
    pull03: f32,
    #[serde(rename = "PULL0.6")]
    pull06: f32,
}

#[derive(Serialize)]
struct TickDrone {
    t: f64,
    p: [f64; 3],
    v: [f64; 3],
    q: [f64; 4],
    w: [f64; 3],
}

#[derive(Serialize)]
struct TickPlate {
    c: [f64; 3],
    q: [f64; 4],
    v: [f64; 3],
    age: f64,
    visible: bool,
}

#[derive(Serialize)]
struct Tick<'a> {
    wall: f64,
    phase: Phase,
    k: usize,
    trial: u64,
    switching: bool,
    rates_only: Option<bool>,
    obs: Option<&'a [f64]>,
    action: Option<[f64; 4]>,
    thrust_norm: Option<f64>,
    rates_frd: Option<[f64; 3]>,
    ekf: TickDrone,
    #[serde(skip_serializing_if = "Option::is_none")]
    plate: Option<TickPlate>,
}

pub struct Snapshot<'a> {
    pub pm: &'a Machine,
    pub out: &'a Output,
    pub drone: Option<Drone>,
    pub plate: Option<Plate>,
    pub rates_only: Option<bool>,
    pub trial: u64,
    pub armed: bool,
    pub offboard: bool,
    pub window: Option<&'a Window>,
}

pub struct Telemetry {
    t_start: f64,
    period: f64,
    diag_period: f64,
    tick_log: Jsonl,
    log_dir: Option<PathBuf>,
    general_log: Jsonl,
    n_ticks: usize,
    last_tick: Option<f64>,
    last_diag: f64,
    last_report: f64,
    cycle: Stats,
    jitter: Stats,
    infer: Stats,
    compute: Stats,
    stats_scratch: Vec<f64>,
    obs_scratch: Vec<f64>,
}

impl Telemetry {
    pub fn new(
        t: f64,
        rate: f64,
        diag_rate: f64,
        tick_log: Option<PathBuf>,
        log_dir: Option<PathBuf>,
        obs_dim: usize,
    ) -> io::Result<Self> {
        if let Some(dir) = &log_dir {
            fs::create_dir_all(dir)?;
        }
        let general_log = Jsonl::create(log_dir.as_ref().map(|dir| dir.join("ticks.jsonl")))?;
        Ok(Self {
            t_start: t,
            period: 1.0 / rate,
            diag_period: 1.0 / diag_rate,
            tick_log: Jsonl::append(tick_log),
            log_dir,
            general_log,
            n_ticks: 0,
            last_tick: None,
            last_diag: f64::NEG_INFINITY,
            last_report: f64::NEG_INFINITY,
            cycle: Stats::new(),
            jitter: Stats::new(),
            infer: Stats::new(),
            compute: Stats::new(),
            stats_scratch: Vec::with_capacity(STATS_WINDOW),
            obs_scratch: Vec::with_capacity(obs_dim),
        })
    }

    pub fn record(&mut self, now: f64, s: Snapshot<'_>) -> io::Result<Option<String>> {
        let Snapshot {
            pm,
            out,
            drone,
            plate,
            rates_only,
            trial,
            armed,
            offboard,
            window,
        } = s;
        let dt = self.last_tick.replace(now).map(|prev| now - prev);
        if let Some(dt) = dt {
            self.cycle.add(dt);
            self.jitter.add((dt - self.period).abs());
        }
        if let Some(infer) = out.infer_s {
            self.infer.add(infer);
        }
        self.n_ticks += 1;

        if self.tick_log.path.is_some()
            && let Some(d) = drone
            && matches!(out.phase, Phase::Perch | Phase::Verify)
        {
            self.obs_scratch.clear();
            if out.action.is_some() {
                self.obs_scratch
                    .extend(pm.observation().iter().map(|&x| rounded(x.into(), 1e5)));
            }
            self.tick_log.write(&Tick {
                wall: rounded(now, 1e5),
                phase: out.phase,
                k: pm.k,
                trial,
                switching: pm.switching(),
                rates_only,
                obs: out.action.map(|_| self.obs_scratch.as_slice()),
                action: out.action.map(|a| round_array(a, 1e5)),
                thrust_norm: out.thrust.map(|v| rounded(v.into(), 1e5)),
                rates_frd: out.rates.map(|r| round_vec(r, 1e5)),
                ekf: TickDrone {
                    t: rounded(d.t, 1e5),
                    p: round_vec(d.p, 1e5),
                    v: round_vec(d.v, 1e5),
                    q: round_array(rot_to_quat(d.r), 1e5),
                    w: round_vec(d.w, 1e5),
                },
                plate: plate.map(|p| TickPlate {
                    c: round_vec(p.c, 1e5),
                    q: round_array(rot_to_quat(p.r), 1e5),
                    v: round_vec(p.v, 1e5),
                    age: rounded(p.age, 1e4),
                    visible: p.visible,
                }),
            })?;
        }
        let publish = now - self.last_diag >= self.diag_period;
        if !publish && self.general_log.path.is_none() {
            return Ok(None);
        }
        let events: [&str; 4] =
            std::array::from_fn(|i| out.events.get(i).map_or("", |e| e.as_str()));
        let mut rec = Diag {
            t: rounded(now - self.t_start, 1e4),
            phase: out.phase,
            attempt: pm.attempt,
            rho: plate.map_or(0.0, |p| rounded(p.reliability.into(), 1e3)),
            plate_age: plate.map(|p| rounded(p.age, 1e3)),
            plate_vis: plate.map_or(0.0, |p| rounded(p.visible_for, 1e2)),
            tag_visible: plate.is_some_and(|p| p.visible),
            action: out.action.map(|a| round_array(a, 1e4)),
            thrust_norm: out.thrust.map(|v| rounded(v.into(), 1e4)),
            rates_frd: out.rates.map(|r| round_vec(r, 1e3)),
            loop_dt_ms: dt.map(|v| rounded(v * 1e3, 1e3)),
            infer_ms: out.infer_s.map(|v| rounded(v * 1e3, 1e4)),
            armed,
            offboard,
            events: &events[..out.events.len()],
            trial,
            ready: pm.ready,
            result: pm.result,
            q2: pm.q2_result.as_ref(),
            p_enu: drone.map(|d| round_vec(d.p, 1e3)),
            plate_c_enu: plate.map(|p| round_vec(p.c, 1e3)),
            plate_v_enu: plate.map(|p| round_vec(p.v, 1e3)),
            q2_probe: None,
            b4e_thresholds: None,
            loop_ms: None,
            jitter_ms: None,
            infer_ms_stats: None,
        };
        self.general_log.write(&rec)?;
        if !publish {
            return Ok(None);
        }
        self.last_diag = now;
        if let Some(v) = &pm.verifier {
            if out.phase == Phase::Verify {
                rec.q2_probe = Some(Probe {
                    macro_name: v.macro_name(),
                    last_point: v.points.last().copied(),
                    slip_mm: window.and_then(Window::slip_mm),
                });
            }
            if v.cfg.rule == Rule::FsmB4e {
                let [relax, pull03, pull06] = v.cfg.slip_off_mm;
                rec.b4e_thresholds = Some(B4eThresholds {
                    relax,
                    pull03,
                    pull06,
                });
            }
        }
        rec.loop_ms = Some(self.cycle.summary(&mut self.stats_scratch));
        rec.jitter_ms = Some(self.jitter.summary(&mut self.stats_scratch));
        rec.infer_ms_stats = Some(self.infer.summary(&mut self.stats_scratch));
        Ok(Some(serde_json::to_string(&rec)?))
    }

    fn timing(&mut self) -> Timings {
        Timings {
            loop_ms: self.cycle.summary(&mut self.stats_scratch),
            jitter_ms: self.jitter.summary(&mut self.stats_scratch),
            infer_ms: self.infer.summary(&mut self.stats_scratch),
            tick_compute_ms: self.compute.summary(&mut self.stats_scratch),
        }
    }

    pub fn tick_time(&mut self, elapsed: Duration) {
        self.compute.add(elapsed.as_secs_f64());
    }

    pub fn report(
        &mut self,
        now: f64,
        machine: &Machine,
        odom: usize,
        meas: usize,
    ) -> io::Result<()> {
        if now - self.last_report >= 5.0 {
            self.last_report = now;
            let mut record = serde_json::to_value(self.timing())?;
            record["phase"] = serde_json::to_value(machine.phase)?;
            record["odom_msgs"] = odom.into();
            record["meas"] = meas.into();
            eprintln!("{record}");
        }
        Ok(())
    }

    pub fn finish(&mut self, now: f64, pm: &Machine, meas: usize) -> io::Result<()> {
        let mut summary = serde_json::to_value(self.timing())?;
        summary["ticks"] = self.n_ticks.into();
        summary["wall_s"] = rounded(now - self.t_start, 1e2).into();
        summary["policy_backend"] = "rust".into();
        summary["phase"] = serde_json::to_value(pm.phase)?;
        summary["result"] = serde_json::to_value(pm.result)?;
        summary["meas"] = meas.into();
        let output = serde_json::to_string_pretty(&summary)?;
        if let Some(dir) = &self.log_dir {
            fs::write(Path::new(dir).join("summary.json"), &output)?;
        }
        println!("{output}");
        Ok(())
    }
}
