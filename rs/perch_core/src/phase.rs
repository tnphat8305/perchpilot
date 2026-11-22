use crate::frames::{ActionScale, Drone, Plate, Vec3, horizontal, observation, wrap_pi, z};
use crate::model::Policy;
use crate::q2::{Decision, Features, Q2Result, Verifier, Window};
use crate::{Action, G};
use arrayvec::{ArrayString, ArrayVec};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, fmt, fmt::Write, io, time::Instant};

const SET_MODE: u16 = 176;
const ARM_DISARM: u16 = 400;
const PERCH_MODE: u16 = 31010;
const FORCE: f32 = 21196.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Phase {
    #[default]
    Wait,
    Approach,
    Perch,
    Verify,
    Rest,
    Depart,
    Abort,
    Done,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Wait => "WAIT",
            Self::Approach => "APPROACH",
            Self::Perch => "PERCH",
            Self::Verify => "VERIFY",
            Self::Rest => "REST",
            Self::Depart => "DEPART",
            Self::Abort => "ABORT",
            Self::Done => "DONE",
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    None,
    Position,
    Rates,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default)]
pub struct PhaseConfig {
    pub approach_back: f32,
    pub approach_up: f32,
    pub approach_tol: f32,
    pub approach_vtol: f32,
    pub approach_yaw_tol: f32,
    pub approach_settle_s: f64,
    pub extra_delay_steps: usize,
    pub yaw_cmd_scale: f32,
    pub switch_timeout_s: f64,
    pub tag_stable_s: f64,
    pub approach_timeout_s: f64,
    pub arm_delay_s: f64,
    pub arm_retry_s: f64,
    pub perch_lost_s: f64,
    pub contact_d: f32,
    pub plate_half: f32,
    pub min_below_plate: f32,
    pub verify_s: f64,
    pub rest_s: f64,
    pub force_disarm_after_s: f64,
    pub force_arm_after_tries: u32,
    pub depart_dist: f32,
    pub depart_timeout_s: f64,
    pub depart_pushoff: bool,
    pub pushoff_g: f32,
    pub pushoff_s: f64,
    pub pushoff_level_s: f64,
    pub pushoff_idle_u: f32,
    pub abort_back: f32,
    pub abort_up: f32,
    pub abort_hold_s: f64,
    pub max_attempts: u32,
    pub max_steps: usize,
    pub standby_offset: Option<[f32; 3]>,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            approach_back: 4.0,
            approach_up: 1.0,
            approach_tol: 0.5,
            approach_vtol: 0.3,
            approach_yaw_tol: 0.25,
            approach_settle_s: 0.3,
            extra_delay_steps: 0,
            yaw_cmd_scale: 1.0,
            switch_timeout_s: 1.0,
            tag_stable_s: 0.3,
            approach_timeout_s: 90.0,
            arm_delay_s: 1.0,
            arm_retry_s: 1.0,
            perch_lost_s: 1.0,
            contact_d: 0.02,
            plate_half: 0.22,
            min_below_plate: 1.0,
            verify_s: 0.5,
            rest_s: 5.0,
            force_disarm_after_s: 1.0,
            force_arm_after_tries: 2,
            depart_dist: 2.0,
            depart_timeout_s: 15.0,
            depart_pushoff: false,
            pushoff_g: 1.5,
            pushoff_s: 0.25,
            pushoff_level_s: 0.6,
            pushoff_idle_u: 0.05,
            abort_back: 1.5,
            abort_up: 1.0,
            abort_hold_s: 3.0,
            max_attempts: 3,
            max_steps: 600,
            standby_offset: None,
        }
    }
}

pub struct Input<'a> {
    pub t: f64,
    pub drone: Option<Drone>,
    pub armed: bool,
    pub offboard: bool,
    pub plate: Option<Plate>,
    pub contact: Option<bool>,
    pub rates_mode: Option<bool>,
    pub window: Option<&'a Window>,
}

#[derive(Default, Debug)]
pub struct Output {
    pub phase: Phase,
    pub mode: Mode,
    pub pos: Option<Vec3>,
    pub vel: Option<Vec3>,
    pub yaw: Option<f32>,
    pub rates: Option<Vec3>,
    pub thrust: Option<f32>,
    pub action: Option<Action>,
    pub infer_s: Option<f64>,
    pub commands: ArrayVec<(u16, f32, f32), 4>,
    pub hold_cmd: Option<f32>,
    pub events: ArrayVec<ArrayString<160>, 4>,
}

impl Output {
    fn event(&mut self, description: fmt::Arguments<'_>) {
        let mut event = ArrayString::new();
        let _ = event.write_fmt(description);
        self.events.push(event);
    }

    fn position(&mut self, p: Vec3, v: Option<Vec3>, yaw: Option<f32>) {
        self.mode = Mode::Position;
        self.pos = Some(p);
        self.vel = v;
        self.yaw = yaw;
    }

    fn rates(&mut self, (rates, thrust): (Vec3, f32)) {
        self.mode = Mode::Rates;
        self.rates = Some(rates);
        self.thrust = Some(thrust);
    }

    pub fn apply_position_handoff(
        &mut self,
        scale: ActionScale,
        drone: Option<Drone>,
        rates_only: Option<bool>,
    ) {
        if self.mode == Mode::Position
            && rates_only == Some(true)
            && let Some(drone) = drone
        {
            let (rates, thrust) = scale.level(drone);
            self.rates = Some(rates);
            self.thrust = Some(thrust);
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
enum Stage {
    #[default]
    Arm,
    Spool(f64),
    Push(f64),
    Level(f64),
    Climb,
}

#[derive(Default)]
struct Flags {
    sent: bool,
    ok_since: Option<f64>,
    switching: bool,
    forced_t: Option<f64>,
    stage: Option<Stage>,
    armed_once: bool,
    wait_for_spool: bool,
    armed_since: Option<f64>,
    abort_init: bool,
}

pub struct Machine {
    pub cfg: PhaseConfig,
    pub scale: ActionScale,
    pub phase: Phase,
    pub verifier: Option<Verifier>,
    pub q2_result: Option<Q2Result>,
    pub attempt: u32,
    pub k: usize,
    pub ready: bool,
    pub result: Option<&'static str>,
    policy: Policy,
    tag_obs: bool,
    obs: Vec<f32>,
    pad_offset: f32,
    time_enter: Option<f64>,
    prev_action: Action,
    history: Vec<Action>,
    delay: VecDeque<Action>,
    last_thr: f32,
    last_acc: f32,
    anchor: Option<Vec3>,
    time_cmd: f64,
    tries: u32,
    flags: Flags,
    standby: Option<Vec3>,
    returning: bool,
}

impl Machine {
    pub fn new(
        cfg: PhaseConfig,
        policy: Policy,
        scale: ActionScale,
        pad_offset: f32,
        phase: Phase,
        verifier: Option<Verifier>,
    ) -> Self {
        let prev_action = scale.hover_action();
        Self {
            cfg,
            scale,
            phase,
            verifier,
            q2_result: None,
            attempt: 1,
            k: 0,
            ready: false,
            result: None,
            tag_obs: policy.meta.has_tag(),
            obs: vec![0.0; policy.meta.obs_dim],
            pad_offset,
            time_enter: None,
            prev_action,
            history: vec![prev_action; policy.meta.history_len()],
            delay: VecDeque::from(vec![prev_action; cfg.extra_delay_steps]),
            last_thr: scale.hover,
            last_acc: G,
            anchor: None,
            time_cmd: f64::NEG_INFINITY,
            tries: 0,
            flags: Flags::default(),
            standby: None,
            returning: false,
            policy,
        }
    }

    pub fn observation(&self) -> &[f32] {
        &self.obs
    }

    pub fn switching(&self) -> bool {
        self.flags.switching
    }

    pub fn start_trial(&mut self, t: f64) -> bool {
        if self.cfg.standby_offset.is_none() || !self.ready || self.phase != Phase::Wait {
            return false;
        }
        self.ready = false;
        self.transition(Phase::Approach, t);
        true
    }

    pub fn reset_trial(&mut self) {
        if self.cfg.standby_offset.is_some() && self.phase != Phase::Wait {
            self.returning = true;
        }
    }

    fn return_to_standby(&mut self, inp: &Input<'_>, d: Drone, out: &mut Output) {
        if self.phase == Phase::Depart {
            return;
        }
        if self.result == Some("depart_timeout") && (!inp.armed || self.contact(inp, d)) {
            self.result = None;
            self.go(
                Phase::Depart,
                inp.t,
                out,
                format_args!("evaluation reset: retry depart"),
            );
            return;
        }
        if self.phase == Phase::Rest
            || (matches!(self.phase, Phase::Perch | Phase::Verify) && self.contact(inp, d))
        {
            self.go(
                Phase::Depart,
                inp.t,
                out,
                format_args!("evaluation reset from plate"),
            );
            return;
        }
        self.returning = false;
        self.result = None;
        self.anchor = None;
        self.attempt = 1;
        self.reset_perch();
        self.q2_result = None;
        self.last_thr = self.scale.hover;
        self.go(
            Phase::Wait,
            inp.t,
            out,
            format_args!("evaluation return to standby"),
        );
        out.hold_cmd = Some(0.0);
        out.commands.push((PERCH_MODE, 0.0, 0.0));
    }

    fn transition(&mut self, phase: Phase, t: f64) {
        self.phase = phase;
        self.time_enter = Some(t);
        self.time_cmd = f64::NEG_INFINITY;
        self.tries = 0;
        self.flags = Flags::default();
    }

    fn go(&mut self, phase: Phase, t: f64, out: &mut Output, why: fmt::Arguments<'_>) {
        out.event(format_args!("{}->{phase}: {why}", self.phase));
        self.transition(phase, t);
    }

    fn arm(&mut self, time: f64, out: &mut Output, force: bool) {
        if time - self.time_cmd >= self.cfg.arm_retry_s {
            self.time_cmd = time;
            self.tries += 1;
            out.commands.push((SET_MODE, 1.0, 6.0));
            out.commands
                .push((ARM_DISARM, 1.0, if force { FORCE } else { 0.0 }));
        }
    }

    fn reset_perch(&mut self) {
        self.k = 0;
        self.prev_action = self.scale.hover_action();
        self.history.fill(self.prev_action);
        for a in &mut self.delay {
            *a = self.prev_action;
        }
    }

    fn approach_point(&self, plate: Plate) -> (Vec3, f32) {
        let f = horizontal(-z(plate.r));
        let target = plate.c - self.cfg.approach_back * f + self.cfg.approach_up * Vec3::z();
        (target, f.y.atan2(f.x))
    }

    pub fn q2_channels(&self) -> [f32; 3] {
        let tw = self.scale.tw_ratio * G;
        if self.phase == Phase::Verify
            && let Some(v) = &self.verifier
            && !v.done
        {
            return v.channels(tw);
        }
        if self.phase == Phase::Rest {
            [0.0, 3.0, 0.0]
        } else {
            [self.last_acc / tw, 0.0, 1.0]
        }
    }

    fn contact(&self, inp: &Input<'_>, d: Drone) -> bool {
        if let Some(contact) = inp.contact {
            return contact;
        }
        let Some(plate) = inp.plate.filter(|p| p.valid) else {
            return false;
        };
        let (u, w, n) = plate.axes();
        let rel = d.p - self.pad_offset * z(d.r) - plate.c;
        (-0.05..=self.cfg.contact_d).contains(&rel.dot(&n))
            && rel.dot(&u).abs().max(rel.dot(&w).abs()) < self.cfg.plate_half
    }

    fn perch(
        &mut self,
        inp: &Input<'_>,
        d: Drone,
        plate: Option<Plate>,
        out: &mut Output,
        el: f64,
    ) {
        let Some(p) = plate.filter(|p| p.age <= self.cfg.perch_lost_s) else {
            self.go(
                Phase::Abort,
                inp.t,
                out,
                format_args!("plate estimate lost"),
            );
            return;
        };
        let why = if self.k >= self.cfg.max_steps {
            Some("perch timeout")
        } else if d.r[(2, 2)] < -0.3 {
            Some("upside down")
        } else if d.p.z < p.c.z - self.cfg.min_below_plate {
            Some("too low")
        } else if !inp.armed {
            Some("disarmed in flight")
        } else if self.flags.switching
            && inp.rates_mode == Some(false)
            && el > self.cfg.switch_timeout_s
        {
            Some("PX4 did not switch to body-rate control")
        } else {
            None
        };
        if let Some(why) = why {
            self.go(Phase::Abort, inp.t, out, format_args!("{why}"));
            return;
        }
        if self.flags.switching && inp.rates_mode != Some(false) {
            self.flags.switching = false;
            out.event(format_args!("PX4 body-rate control after {el:.3} s"));
        }
        if self.flags.switching {
            out.rates(self.scale.level(d));
            let (target, yaw) = self.approach_point(p);
            out.pos = Some(target);
            out.yaw = Some(yaw);
            out.vel = Some(p.v);
        } else if self.k > 0 && self.contact(inp, d) {
            let k = self.k;
            self.go(
                Phase::Verify,
                inp.t,
                out,
                format_args!("contact after {k} steps"),
            );
            if let Some(v) = &mut self.verifier {
                v.reset(inp.t, self.last_acc);
                self.q2_result = None;
            }
        } else {
            observation(
                d,
                p,
                self.prev_action,
                self.k as f32 / self.cfg.max_steps as f32,
                self.pad_offset,
                self.tag_obs,
                &mut self.obs,
            );
            for (slot, action) in self.obs[28..].chunks_exact_mut(4).zip(&self.history) {
                slot.copy_from_slice(action);
            }
            let start = Instant::now();
            let action = self.policy.infer(&self.obs);
            out.infer_s = Some(start.elapsed().as_secs_f64());
            let mut sent = if let Some(old) = self.delay.pop_front() {
                self.delay.push_back(action);
                old
            } else {
                action
            };
            sent[3] *= self.cfg.yaw_cmd_scale;
            out.rates(self.scale.action(sent));
            out.action = Some(action);
            if !self.history.is_empty() {
                self.history.rotate_right(1);
                self.history[0] = self.prev_action;
            }
            self.prev_action = action;
            self.last_thr = out.thrust.unwrap();
            self.last_acc = self.scale.acc(sent[0]);
            self.k += 1;
        }
    }

    fn verify(&mut self, inp: &Input<'_>, out: &mut Output) -> io::Result<()> {
        let v = self.verifier.as_mut().unwrap();
        let result = v.step(inp.t, inp.contact, inp.window)?;
        out.rates((Vec3::zeros(), self.scale.acc_to_u(v.thrust).clamp(0.0, 1.0)));
        if let Some(mut result) = result {
            let reason = result.why;
            if result.decision == Decision::Off && inp.contact == Some(false) {
                result.decision = Decision::SelfAbort;
                result.why = "contact lost before MOTORS_OFF";
            }
            let decision = result.decision;
            self.q2_result = Some(result);
            if decision == Decision::Off {
                self.go(
                    Phase::Rest,
                    inp.t,
                    out,
                    format_args!("Q2 {reason}: MOTORS_OFF"),
                );
            } else {
                self.go(
                    Phase::Depart,
                    inp.t,
                    out,
                    format_args!("Q2 {}: {reason}", decision.as_str()),
                );
                out.hold_cmd = Some(0.0);
                self.flags.stage = Some(if decision == Decision::Abort {
                    Stage::Push(inp.t)
                } else {
                    Stage::Level(inp.t)
                });
            }
        }
        Ok(())
    }

    fn check_departed(
        &mut self,
        inp: &Input<'_>,
        d: Drone,
        plate: Option<Plate>,
        out: &mut Output,
    ) {
        let anchor = self.anchor.unwrap();
        out.position(anchor, None, None);
        let dist = plate.map(|p| (d.p - p.c).norm());
        if (d.p - anchor).norm() < 0.5 && dist.is_none_or(|x| x >= self.cfg.depart_dist) {
            self.result = Some("departed");
            if let Some(dist) = dist {
                self.go(
                    Phase::Done,
                    inp.t,
                    out,
                    format_args!("departed, {dist:.2} m from plate"),
                );
            } else {
                self.go(
                    Phase::Done,
                    inp.t,
                    out,
                    format_args!("departed, nan m from plate"),
                );
            }
        }
    }

    fn depart(
        &mut self,
        inp: &Input<'_>,
        d: Drone,
        plate: Option<Plate>,
        out: &mut Output,
        el: f64,
    ) {
        let cfg = self.cfg;
        let force = cfg.standby_offset.is_none() && self.tries >= cfg.force_arm_after_tries;
        let normal = plate.map_or_else(Vec3::z, |p| z(p.r));
        if cfg.depart_pushoff || self.flags.stage.is_some() {
            let now = inp.t;
            match self.flags.stage.unwrap_or_default() {
                Stage::Arm => {
                    out.rates((Vec3::zeros(), cfg.pushoff_idle_u));
                    if inp.armed && inp.offboard && inp.rates_mode != Some(false) {
                        self.flags.stage = Some(Stage::Spool(now));
                    } else {
                        self.arm(now, out, force);
                    }
                }
                Stage::Spool(since) => {
                    out.rates((Vec3::zeros(), cfg.pushoff_idle_u));
                    if !inp.armed {
                        self.flags.stage = Some(Stage::Arm);
                    } else if now - since >= 1.0 {
                        out.commands.push((PERCH_MODE, 0.0, 0.0));
                        out.hold_cmd = Some(0.0);
                        self.flags.stage = Some(Stage::Push(now));
                        out.event(format_args!("depart: released, push-off along body z"));
                    }
                }
                Stage::Push(since) => {
                    out.rates((
                        Vec3::zeros(),
                        self.scale.acc_to_u(cfg.pushoff_g * G).clamp(0.0, 1.0),
                    ));
                    if now - since >= cfg.pushoff_s {
                        self.flags.stage = Some(Stage::Level(now));
                    }
                }
                Stage::Level(since) => {
                    out.rates(self.scale.level(d));
                    if now - since >= cfg.pushoff_level_s {
                        self.anchor = Some(d.p + cfg.depart_dist * horizontal(normal) + Vec3::z());
                        self.flags.stage = Some(Stage::Climb);
                        out.event(format_args!("depart: levelled, position control"));
                    }
                }
                Stage::Climb => {}
            }
            if self.flags.stage == Some(Stage::Climb) {
                self.check_departed(inp, d, plate, out);
            }
        } else if self.flags.armed_once {
            self.check_departed(inp, d, plate, out);
        } else {
            out.position(d.p, None, None);
            if inp.armed && inp.offboard {
                let since = *self.flags.armed_since.get_or_insert(inp.t);
                if !self.flags.wait_for_spool || inp.t - since >= 1.0 {
                    self.flags.armed_once = true;
                    out.commands.push((PERCH_MODE, 0.0, 0.0));
                    out.hold_cmd = Some(0.0);
                    self.anchor = Some(d.p + cfg.depart_dist * normal);
                }
            } else {
                self.flags.wait_for_spool |= cfg.standby_offset.is_some() && !inp.armed;
                self.flags.armed_since = None;
                self.arm(inp.t, out, force);
            }
        }
        if el > cfg.depart_timeout_s && self.phase == Phase::Depart {
            self.result = Some("depart_timeout");
            self.go(Phase::Done, inp.t, out, format_args!("depart timeout"));
        }
    }

    pub fn step(&mut self, inp: Input<'_>) -> io::Result<Output> {
        let mut out = Output {
            phase: self.phase,
            ..Output::default()
        };
        self.time_enter.get_or_insert(inp.t);
        let Some(d) = inp.drone else {
            self.ready = false;
            return Ok(out);
        };
        let c = self.cfg;
        let plate = inp.plate.filter(|p| p.valid);
        if self.returning {
            self.return_to_standby(&inp, d, &mut out);
        }
        let el = inp.t - self.time_enter.unwrap();
        match self.phase {
            Phase::Wait => {
                if let Some(offset) = c.standby_offset {
                    if self.standby.is_none() {
                        self.standby = Some(d.p + Vec3::from(offset));
                        self.time_enter = Some(inp.t);
                    }
                    let standby = self.standby.unwrap();
                    out.position(standby, None, Some(0.0));
                    self.ready = inp.armed
                        && inp.offboard
                        && (d.p - standby).norm() < 0.30
                        && d.v.norm() < 0.35;
                    if (!inp.armed || !inp.offboard)
                        && inp.t - self.time_enter.unwrap() >= c.arm_delay_s
                    {
                        self.arm(inp.t, &mut out, false);
                    }
                } else {
                    out.position(d.p, None, None);
                    if plate.is_some() {
                        self.go(
                            Phase::Approach,
                            inp.t,
                            &mut out,
                            format_args!("plate estimate valid"),
                        );
                    }
                }
            }
            Phase::Approach => {
                if let Some(p) = plate {
                    let (target, yaw) = self.approach_point(p);
                    self.anchor = Some(target);
                    out.position(target, Some(p.v), Some(yaw));
                    let err = (d.p - target).norm();
                    let verr = (d.v - p.v).norm();
                    let yerr = wrap_pi(d.r[(1, 0)].atan2(d.r[(0, 0)]) - yaw).abs();
                    let ready = inp.armed
                        && inp.offboard
                        && err < c.approach_tol
                        && verr < c.approach_vtol
                        && yerr < c.approach_yaw_tol
                        && p.visible_for >= c.tag_stable_s;
                    if !ready {
                        self.flags.ok_since = None;
                    } else if inp.t - *self.flags.ok_since.get_or_insert(inp.t)
                        >= c.approach_settle_s
                    {
                        self.reset_perch();
                        self.go(
                            Phase::Perch,
                            inp.t,
                            &mut out,
                            format_args!(
                                "at approach point err={err:.2} verr={verr:.2} yerr={yerr:.2}"
                            ),
                        );
                        self.flags.switching = true;
                    }
                } else {
                    out.position(self.anchor.unwrap_or(d.p), None, None);
                }
                if (!inp.armed || !inp.offboard) && el >= c.arm_delay_s {
                    self.arm(inp.t, &mut out, false);
                }
                if el > c.approach_timeout_s && self.phase == Phase::Approach {
                    self.go(
                        Phase::Abort,
                        inp.t,
                        &mut out,
                        format_args!("approach timeout"),
                    );
                }
            }
            Phase::Perch => self.perch(&inp, d, plate, &mut out, el),
            _ => {}
        }

        let el = inp.t - self.time_enter.unwrap();
        match self.phase {
            Phase::Verify if self.verifier.is_some() => self.verify(&inp, &mut out)?,
            Phase::Verify => {
                out.rates((Vec3::zeros(), self.last_thr));
                if el >= c.verify_s {
                    if self.contact(&inp, d) {
                        self.go(
                            Phase::Rest,
                            inp.t,
                            &mut out,
                            format_args!("verify placeholder: contact held"),
                        );
                    } else {
                        self.go(
                            Phase::Abort,
                            inp.t,
                            &mut out,
                            format_args!("verify placeholder: contact lost"),
                        );
                    }
                }
            }
            Phase::Rest if inp.armed && inp.contact == Some(false) && self.verifier.is_some() => {
                self.go(
                    Phase::Depart,
                    inp.t,
                    &mut out,
                    format_args!("REST: contact lost while armed"),
                );
                let reason = "contact lost after MOTORS_OFF was commanded, before disarm";
                let result = self.q2_result.get_or_insert(Q2Result {
                    decision: Decision::SelfAbort,
                    why: reason,
                    d: None,
                    macro_name: None,
                    t: None,
                    features: Features::default(),
                    points: None,
                });
                result.decision = Decision::SelfAbort;
                result.why = reason;
                out.hold_cmd = Some(0.0);
                self.flags.stage = Some(Stage::Level(inp.t));
            }
            Phase::Rest => {
                out.position(d.p, None, None);
                if !self.flags.sent {
                    out.commands.push((PERCH_MODE, 1.0, 0.0));
                    out.commands.push((ARM_DISARM, 0.0, 0.0));
                    self.flags.sent = true;
                } else if inp.armed
                    && el >= c.force_disarm_after_s
                    && inp.contact != Some(false)
                    && self
                        .flags
                        .forced_t
                        .is_none_or(|last| inp.t - last >= c.arm_retry_s)
                {
                    out.commands.push((ARM_DISARM, 0.0, FORCE));
                    if self.flags.forced_t.is_none() {
                        out.event(format_args!("plain disarm rejected -> forced disarm"));
                    }
                    self.flags.forced_t = Some(inp.t);
                }
                if c.standby_offset.is_none() && el >= c.rest_s {
                    self.go(
                        Phase::Depart,
                        inp.t,
                        &mut out,
                        format_args!("rest time over"),
                    );
                }
            }
            Phase::Depart => self.depart(&inp, d, plate, &mut out, el),
            Phase::Abort => {
                if !self.flags.abort_init {
                    self.flags.abort_init = true;
                    let n = plate.map_or_else(Vec3::z, |p| z(p.r));
                    self.anchor = Some(d.p + c.abort_back * horizontal(n) + c.abort_up * Vec3::z());
                    out.hold_cmd = Some(0.0);
                }
                out.position(self.anchor.unwrap(), None, None);
                if c.standby_offset.is_none() {
                    if !inp.armed && el >= c.arm_delay_s {
                        self.arm(inp.t, &mut out, self.tries >= c.force_arm_after_tries);
                    }
                    if el >= c.abort_hold_s {
                        if self.attempt < c.max_attempts {
                            self.attempt += 1;
                            let attempt = self.attempt;
                            self.go(
                                Phase::Approach,
                                inp.t,
                                &mut out,
                                format_args!("retry, attempt {attempt}"),
                            );
                        } else {
                            self.result = Some("aborted");
                            self.go(
                                Phase::Done,
                                inp.t,
                                &mut out,
                                format_args!("max attempts reached"),
                            );
                        }
                    }
                }
            }
            Phase::Done => {
                out.position(*self.anchor.get_or_insert(d.p), None, None);
            }
            _ => {}
        }
        out.phase = self.phase;
        Ok(out)
    }
}
