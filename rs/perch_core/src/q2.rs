use crate::frames::{Vec3, frd_flu};
use crate::model::{Belief, C5_ROWS, C5_V1, C5_V2, C5Row, C5Samples};
use crate::perception::SENSOR_TIMEOUT;
use crate::{G, invalid};
use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, io, path::Path};

const SCHEDULE: [&str; 4] = ["RELAX", "PULL0.3", "PULL0.6", "PULL0.9"];

#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    #[default]
    Placeholder,
    None,
    Dthr,
    FsmB4b,
    FsmB4e,
    Probe,
}

impl Rule {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Placeholder => "placeholder",
            Self::None => "none",
            Self::Dthr => "dthr",
            Self::FsmB4b => "fsm_b4b",
            Self::FsmB4e => "fsm_b4e",
            Self::Probe => "probe",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default)]
pub struct Q2Config {
    pub rule: Rule,
    pub tau: f32,
    pub s: f32,
    pub d_min: usize,
    pub e_thr: f32,
    pub slip_off_mm: [f32; 3],
}

impl Default for Q2Config {
    fn default() -> Self {
        Self {
            rule: Rule::Placeholder,
            tau: 0.56,
            s: 0.6,
            d_min: 0,
            e_thr: 0.657_188_95,
            slip_off_mm: [0.0; 3],
        }
    }
}

impl Q2Config {
    pub fn load(rule: Rule, decide: &Path, key: &str) -> io::Result<Self> {
        let data: Value = serde_json::from_slice(&fs::read(decide)?)?;
        let chosen = &data["chosen_on_val"];
        let threshold = |section: &str, field: &str| {
            chosen[section][field]
                .as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| invalid(format!("missing chosen_on_val.{section}.{field}")))
        };
        Ok(Self {
            rule,
            tau: threshold(key, "tau")?,
            s: threshold(key, "s")?,
            d_min: usize::from(key.contains("minPULL0.3")),
            e_thr: threshold("B4b_FSM_probe", "e_thr")?,
            ..Self::default()
        })
    }

    pub fn schedule(self) -> &'static [&'static str] {
        &SCHEDULE[..if self.rule == Rule::Probe { 4 } else { 3 }]
    }
}

#[derive(Clone)]
pub struct Window {
    pub samples: C5Samples,
    pub channels: usize,
}

impl Window {
    pub fn slip_mm(&self) -> Option<f32> {
        if self.channels != C5_V2 {
            return None;
        }
        let (u, w) = self
            .samples
            .iter()
            .fold((0.0, 0.0), |(u, w), row| (u + row[11], w + row[12]));
        Some(u.hypot(w))
    }

    fn hp_energy(&self, d: usize) -> f32 {
        let len = if d == 0 { 15 } else { 50 };
        self.samples[C5_ROWS - len..]
            .windows(2)
            .map(|x| (x[1][0] - x[0][0]).powi(2) + (x[1][1] - x[0][1]).powi(2))
            .sum::<f32>()
            / (len - 1) as f32
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Imu {
    pub acc: [f32; 3],
    pub gyro: [f32; 3],
}

pub struct Sensors {
    imu: Option<Imu>,
    contact: Option<(f64, bool)>,
    normal: f32,
    slip: Option<[f32; 2]>,
    rows: C5Samples,
    next: usize,
    len: usize,
    channels: usize,
}

impl Default for Sensors {
    fn default() -> Self {
        Self {
            imu: None,
            contact: None,
            normal: 0.0,
            slip: None,
            rows: [[0.0; C5_V2]; C5_ROWS],
            next: 0,
            len: 0,
            channels: C5_V1,
        }
    }
}

impl Sensors {
    pub fn imu(&mut self, imu: Imu) {
        self.imu = Some(imu);
    }

    pub fn hold(&mut self, data: &[f64], now: f64) {
        if !matches!(data.len(), 3 | 5) {
            return;
        }
        self.contact = Some((now, data[1] == 1.0));
        self.normal = data[2] as f32;
        if data.len() == 5 {
            let sum = self.slip.get_or_insert([0.0, 0.0]);
            sum[0] += data[3] as f32;
            sum[1] += data[4] as f32;
        } else {
            self.slip = None;
        }
    }

    pub fn contact(&self, now: f64) -> Option<bool> {
        self.contact
            .filter(|(t, _)| now - t < SENSOR_TIMEOUT)
            .map(|(_, c)| c)
    }

    pub fn sample(&mut self, now: f64, [thrust, phase, level]: [f32; 3]) -> Option<&[f32]> {
        let imu = self.imu?;
        let mut row: C5Row = [0.0; C5_V2];
        row[..3].copy_from_slice(frd_flu(Vec3::from(imu.acc)).as_slice());
        row[3..6].copy_from_slice(frd_flu(Vec3::from(imu.gyro)).as_slice());
        row[6..11].copy_from_slice(&[
            self.normal,
            f32::from(self.contact(now).unwrap_or(false)),
            thrust,
            phase,
            level,
        ]);
        let channels = if let Some(slip) = &mut self.slip {
            row[11..].copy_from_slice(slip);
            *slip = [0.0; 2];
            C5_V2
        } else {
            C5_V1
        };
        if channels != self.channels {
            self.len = 0;
            self.channels = channels;
        }
        let i = self.next;
        self.rows[i] = row;
        self.next = (i + 1) % C5_ROWS;
        self.len = (self.len + 1).min(C5_ROWS);
        Some(&self.rows[i][..channels])
    }

    pub fn window(&self) -> Option<Window> {
        (self.len == C5_ROWS).then(|| Window {
            samples: std::array::from_fn(|i| self.rows[(self.next + i) % C5_ROWS]),
            channels: self.channels,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Decision {
    Off,
    Abort,
    SelfAbort,
}

impl Decision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "OFF",
            Self::Abort => "ABORT",
            Self::SelfAbort => "SELF_ABORT",
        }
    }
}

// Serialize the decision features as the same optional flat fields as Python.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Features {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_hold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_slip: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hp_energy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slip_mm: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DecisionPoint {
    pub d: usize,
    #[serde(rename = "macro")]
    pub macro_name: &'static str,
    pub t: f64,
    #[serde(flatten)]
    pub features: Features,
}

#[derive(Clone, Debug, Serialize)]
pub struct Q2Result {
    pub decision: Decision,
    pub why: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub d: Option<usize>,
    #[serde(rename = "macro", skip_serializing_if = "Option::is_none")]
    pub macro_name: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t: Option<f64>,
    #[serde(flatten)]
    pub features: Features,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub points: Option<ArrayVec<DecisionPoint, 4>>,
}

pub fn rounded(x: f64, scale: f64) -> f64 {
    (x * scale).round_ties_even() / scale
}

pub struct Verifier {
    pub cfg: Q2Config,
    belief: Option<Belief>,
    t0: f64,
    initial: f32,
    i: usize,
    no_contact_since: Option<f64>,
    pub thrust: f32,
    pub done: bool,
    pub points: ArrayVec<DecisionPoint, 4>,
}

impl Verifier {
    pub fn new(cfg: Q2Config, belief: Option<Belief>) -> io::Result<Self> {
        if cfg.rule == Rule::Placeholder || (cfg.rule == Rule::Dthr && belief.is_none()) {
            return Err(invalid(
                "Q2 rule requires a verifier and dthr requires belief weights",
            ));
        }
        Ok(Self {
            cfg,
            belief,
            t0: 0.0,
            initial: G,
            i: 0,
            no_contact_since: None,
            thrust: G,
            done: false,
            points: ArrayVec::new(),
        })
    }

    pub fn reset(&mut self, time: f64, thrust: f32) {
        self.t0 = time;
        self.initial = thrust;
        self.i = 0;
        self.no_contact_since = None;
        self.thrust = thrust;
        self.done = false;
        self.points.clear();
    }

    pub fn macro_name(&self) -> &'static str {
        self.cfg.schedule()[self.i]
    }

    pub fn channels(&self, tw_g: f32) -> [f32; 3] {
        [
            self.thrust / tw_g,
            if self.i == 0 { 1.0 } else { 2.0 },
            self.level(),
        ]
    }

    fn level(&self) -> f32 {
        [0.1, 0.3, 0.6, 0.9][self.i]
    }

    fn finish(
        &mut self,
        decision: Decision,
        why: &'static str,
        point: Option<DecisionPoint>,
    ) -> Q2Result {
        self.done = true;
        Q2Result {
            decision,
            why,
            d: Some(self.i),
            macro_name: point.map(|p| p.macro_name),
            t: point.map(|p| p.t),
            features: point.map(|p| p.features).unwrap_or_default(),
            points: Some(self.points.clone()),
        }
    }

    pub fn step(
        &mut self,
        now: f64,
        contact: Option<bool>,
        window: Option<&Window>,
    ) -> io::Result<Option<Q2Result>> {
        if self.done {
            return Ok(None);
        }
        if self.cfg.rule == Rule::None {
            return Ok(Some(self.finish(
                Decision::Off,
                "rule none: motors off at touchdown",
                None,
            )));
        }
        if contact == Some(false) {
            if now - *self.no_contact_since.get_or_insert(now) >= 0.03 {
                return Ok(Some(self.finish(
                    Decision::SelfAbort,
                    "contact lost while powered",
                    None,
                )));
            }
        } else {
            self.no_contact_since = None;
        }
        let elapsed = now - self.t0;
        let duration = if self.i == 0 { 0.15 } else { 0.25 };
        if elapsed < duration - 1e-9 {
            let idle = 0.1 * G;
            let target = self.level() * G;
            self.thrust = if self.i > 0 && elapsed > 0.2 {
                target + (idle - target) * ((elapsed - 0.2) / 0.05).clamp(0.0, 1.0) as f32
            } else {
                self.initial + (target - self.initial) * (elapsed / 0.05).clamp(0.0, 1.0) as f32
            };
            return Ok(None);
        }

        let last = self.i + 1 == self.cfg.schedule().len();
        let mut point = DecisionPoint {
            d: self.i,
            macro_name: self.macro_name(),
            t: rounded(elapsed, 1e4),
            features: Features::default(),
        };
        let need_window =
            || window.ok_or_else(|| invalid("Q2 decision requires a full C5 sensor window"));
        let decision = match self.cfg.rule {
            Rule::Dthr => {
                let win = need_window()?;
                let belief = self.belief.as_mut().unwrap();
                if win.channels < belief.channels {
                    return Err(invalid(
                        "belief checkpoint requires C5 v2 optical slip samples",
                    ));
                }
                let [p_hold, p_slip, _] = belief.infer(&win.samples);
                point.features.p_hold = Some(rounded(p_hold.into(), 1e4));
                point.features.p_slip = Some(rounded(p_slip.into(), 1e4));
                if p_slip >= self.cfg.s {
                    Some(Decision::Abort)
                } else if p_hold >= self.cfg.tau && self.i >= self.cfg.d_min {
                    Some(Decision::Off)
                } else {
                    last.then_some(Decision::Abort)
                }
            }
            Rule::FsmB4b => {
                let e = need_window()?.hp_energy(self.i);
                point.features.hp_energy = Some(rounded(e.into(), 1e5));
                if e >= self.cfg.e_thr {
                    Some(Decision::Abort)
                } else {
                    last.then_some(Decision::Off)
                }
            }
            Rule::FsmB4e => {
                let slip = need_window()?
                    .slip_mm()
                    .ok_or_else(|| invalid("fsm_b4e requires C5 v2"))?;
                point.features.slip_mm = Some(rounded(slip.into(), 1e4));
                if slip < self.cfg.slip_off_mm[self.i] {
                    Some(Decision::Off)
                } else {
                    last.then_some(Decision::Abort)
                }
            }
            Rule::Probe => {
                point.features.hp_energy = window.map(|w| rounded(w.hp_energy(self.i).into(), 1e5));
                last.then_some(Decision::Off)
            }
            _ => unreachable!(),
        };
        self.points.push(point);
        if let Some(d) = decision {
            let why = match self.cfg.rule {
                Rule::Dthr => ["dthr at RELAX", "dthr at PULL0.3", "dthr at PULL0.6", ""],
                Rule::FsmB4b => [
                    "fsm_b4b at RELAX",
                    "fsm_b4b at PULL0.3",
                    "fsm_b4b at PULL0.6",
                    "",
                ],
                Rule::FsmB4e => [
                    "fsm_b4e at RELAX",
                    "fsm_b4e at PULL0.3",
                    "fsm_b4e at PULL0.6",
                    "",
                ],
                Rule::Probe => [
                    "probe at RELAX",
                    "probe at PULL0.3",
                    "probe at PULL0.6",
                    "probe at PULL0.9",
                ],
                _ => unreachable!(),
            };
            return Ok(Some(self.finish(d, why[self.i], Some(point))));
        }
        self.i += 1;
        self.t0 = now;
        self.initial = self.thrust;
        Ok(None)
    }
}
