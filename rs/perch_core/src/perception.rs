use crate::frames::{Drone, Mat3, Odometry, Plate, Vec3, quat_to_rot, rot_y};
use serde::Deserialize;
use std::collections::VecDeque;

pub const TAG_PERIOD: f64 = 1.0 / 30.0;
pub const SENSOR_TIMEOUT: f64 = 0.2;

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct TagPose {
    pub position: [f32; 3],
    pub q: [f32; 4],
}

pub struct Perception {
    odometry: VecDeque<Drone>,
    plate: PlateFilter,
    q_acc: f32,
}

impl Perception {
    pub fn new(q_acc: f32) -> Self {
        Self {
            odometry: VecDeque::with_capacity(50),
            plate: PlateFilter::default(),
            q_acc,
        }
    }

    pub fn reset_plate(&mut self) {
        self.plate = PlateFilter::default();
    }

    pub fn odometry(&mut self, msg: Odometry, time: f64) -> bool {
        let Some(drone) = msg.drone(time) else {
            return false;
        };
        if self.odometry.len() == 50 {
            self.odometry.pop_front();
        }
        self.odometry.push_back(drone);
        true
    }

    pub fn drone(&self, now: f64) -> Option<Drone> {
        self.odometry
            .back()
            .copied()
            .filter(|d| now - d.t <= SENSOR_TIMEOUT)
    }

    pub fn tag(&mut self, msg: TagPose, now: f64, age: f64) -> bool {
        let t = now - age;
        if !(0.0..=SENSOR_TIMEOUT).contains(&age)
            || self.drone(now).is_none()
            || self.plate.time.is_some_and(|last| t <= last)
            || !msg.position.into_iter().chain(msg.q).all(f32::is_finite)
            || msg.q.iter().map(|x| x * x).sum::<f32>() < 1e-12
        {
            return false;
        }
        let drone = self
            .odometry
            .iter()
            .min_by(|a, b| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
            .unwrap();
        let camera = drone.r * rot_y(15_f32.to_radians());
        let p: Vec3 = msg.position.into();
        self.plate.update(
            t,
            drone.p + camera * p,
            camera * quat_to_rot(msg.q),
            p.norm(),
            self.q_acc,
        );
        true
    }

    pub fn freeze_velocity(&mut self, freeze: bool) {
        self.plate.freeze = freeze;
    }

    pub fn plate(&self, now: f64) -> Option<Plate> {
        self.plate.estimate(now)
    }
}

#[derive(Default)]
struct PlateFilter {
    c: Vec3,
    v: Vec3,
    r: Mat3,
    time: Option<f64>,
    streak: f64,
    freeze: bool,
    // The process noise and observation covariance are diagonal across spatial axes.
    // Each axis therefore shares the same independent (position, velocity) covariance.
    pp: f32,
    pv: f32,
    vv: f32,
}

impl PlateFilter {
    fn update(&mut self, t: f64, c: Vec3, r: Mat3, distance: f32, q_acc: f32) {
        let noise = (0.01 + 0.01 * distance).powi(2);
        if let Some(last) = self.time {
            let dt = (t - last) as f32;
            self.c += dt * self.v;
            let pp = self.pp + 2.0 * dt * self.pv + dt * dt * self.vv + q_acc * dt.powi(3) / 3.0;
            let pv = self.pv + dt * self.vv + q_acc * dt * dt / 2.0;
            let vv = self.vv + q_acc * dt;
            let kp = pp / (pp + noise);
            let kv = if self.freeze { 0.0 } else { pv / (pp + noise) };
            let res = c - self.c;
            self.c += kp * res;
            self.v += kv * res;
            // Joseph update with a zero velocity gain while frozen.
            let a = 1.0 - kp;
            self.pp = a * a * pp + noise * kp * kp;
            self.pv = a * (pv - kv * pp) + noise * kp * kv;
            self.vv = vv - 2.0 * kv * pv + kv * kv * (pp + noise);
            let svd = (0.7 * self.r + 0.3 * r).svd(true, true);
            let mut u = svd.u.unwrap();
            let vt = svd.v_t.unwrap();
            if (u * vt).determinant() < 0.0 {
                u.column_mut(2).scale_mut(-1.0);
            }
            self.r = u * vt;
        } else {
            self.c = c;
            self.r = r;
            self.pp = noise;
            self.vv = 9.0;
        }
        if self.time.is_none_or(|last| t - last > SENSOR_TIMEOUT) {
            self.streak = t;
        }
        self.time = Some(t);
    }

    fn estimate(&self, t: f64) -> Option<Plate> {
        let last = self.time?;
        let age = t - last;
        let loss = (age - TAG_PERIOD).clamp(0.0, 2.0) as f32;
        Some(Plate {
            c: self.c + age.max(0.0) as f32 * self.v,
            r: self.r,
            v: self.v,
            age,
            valid: age <= 2.0,
            visible_for: if age <= SENSOR_TIMEOUT {
                last - self.streak
            } else {
                0.0
            },
            reliability: (-2.0 * loss).exp(),
            visible: loss == 0.0,
            loss_s: loss,
        })
    }
}
