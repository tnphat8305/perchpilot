use crate::{Action, G};
use nalgebra::{Quaternion, Rotation3, UnitQuaternion};
use serde::Deserialize;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

pub type Vec3 = nalgebra::Vector3<f32>;
pub type Mat3 = nalgebra::Matrix3<f32>;

pub fn ned_enu(v: Vec3) -> Vec3 {
    Vec3::new(v.y, v.x, -v.z)
}

pub fn frd_flu(v: Vec3) -> Vec3 {
    Vec3::new(v.x, -v.y, -v.z)
}

pub fn quat_to_rot([w, x, y, z]: [f32; 4]) -> Mat3 {
    UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z))
        .to_rotation_matrix()
        .into_inner()
}

pub fn rot_to_quat(r: Mat3) -> [f32; 4] {
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r)).into_inner();
    let sign = if q.w < 0.0 { -1.0 } else { 1.0 };
    [q.w, q.i, q.j, q.k].map(|x| x * sign)
}

// Both frame changes are self-inverse. Apply T on the left and F on the right.
pub fn ned_frd_enu_flu(r: Mat3) -> Mat3 {
    Mat3::from_columns(&[
        ned_enu(r.column(0).into_owned()),
        -ned_enu(r.column(1).into_owned()),
        -ned_enu(r.column(2).into_owned()),
    ])
}

pub fn rot_y(angle: f32) -> Mat3 {
    Rotation3::from_axis_angle(&Vec3::y_axis(), angle).into_inner()
}

pub fn wrap_pi(a: f32) -> f32 {
    (a + PI).rem_euclid(TAU) - PI
}

pub fn yaw_enu_ned(a: f32) -> f32 {
    wrap_pi(FRAC_PI_2 - a)
}

pub(crate) fn horizontal(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, 0.0)
        .try_normalize(1e-6)
        .unwrap_or_else(Vec3::x)
}

pub(crate) fn z(r: Mat3) -> Vec3 {
    r.column(2).into_owned()
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Odometry {
    pub position: [f32; 3],
    pub q: [f32; 4],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub pose_frame: u8,
    pub velocity_frame: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Drone {
    pub p: Vec3,
    pub v: Vec3,
    pub r: Mat3,
    pub w: Vec3,
    pub t: f64,
}

impl Odometry {
    pub fn drone(self, t: f64) -> Option<Drone> {
        if self.pose_frame != 1
            || !self
                .position
                .into_iter()
                .chain(self.q)
                .chain(self.velocity)
                .chain(self.angular_velocity)
                .all(f32::is_finite)
            || self.q.iter().map(|x| x * x).sum::<f32>() < 0.25
        {
            return None;
        }
        let r = quat_to_rot(self.q);
        let v = self.velocity.into();
        let v = match self.velocity_frame {
            1 | 2 => v,
            3 => r * v,
            _ => return None,
        };
        Some(Drone {
            p: ned_enu(self.position.into()),
            v: ned_enu(v),
            r: ned_frd_enu_flu(r),
            w: frd_flu(self.angular_velocity.into()),
            t,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Plate {
    pub c: Vec3,
    pub r: Mat3,
    pub v: Vec3,
    pub valid: bool,
    pub age: f64,
    pub visible_for: f64,
    pub reliability: f32,
    pub visible: bool,
    pub loss_s: f32,
}

impl Plate {
    pub fn axes(&self) -> (Vec3, Vec3, Vec3) {
        let u = self.r.column(1).into_owned();
        let n = z(self.r);
        (u, n.cross(&u), n)
    }
}

pub(crate) fn observation(
    d: Drone,
    p: Plate,
    prev: Action,
    clock: f32,
    pad_offset: f32,
    tag: bool,
    out: &mut [f32],
) {
    let (mut u, mut w, n) = p.axes();
    if tag {
        u = n.cross(&Vec3::z());
        u /= u.norm().max(1e-6);
        w = n.cross(&u);
        let suffix = out.len() - 3;
        out[suffix..].copy_from_slice(&[p.reliability, f32::from(p.visible), p.loss_s / 2.0]);
    }
    let axes = Mat3::from_columns(&[u, w, n]).transpose();
    out[..3].copy_from_slice((axes * (d.p - pad_offset * z(d.r) - p.c)).as_slice());
    out[3..6].copy_from_slice((axes * (d.v - p.v)).as_slice());
    let body = axes * d.r;
    for i in 0..3 {
        for j in 0..3 {
            out[6 + 3 * i + j] = body[(i, j)];
        }
    }
    out[15..18].copy_from_slice(d.w.as_slice());
    out[18..21].copy_from_slice(axes.column(2).as_slice());
    out[21..25].copy_from_slice(&prev);
    out[25..28].copy_from_slice(&[clock, p.v.norm(), n.z.clamp(-1.0, 1.0).acos()]);
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThrustModel {
    #[default]
    Linear,
    #[serde(rename = "x500_gz")]
    X500,
    #[serde(rename = "x500_perch_gz")]
    X500Perch,
}

#[derive(Clone, Copy, Debug)]
pub struct ActionScale {
    pub rate_max: Vec3,
    pub tw_ratio: f32,
    pub hover: f32,
    pub model: ThrustModel,
}

impl ActionScale {
    fn motor_kf(self) -> f32 {
        match self.model {
            ThrustModel::X500Perch => 1.144_003e-5,
            _ => 8.548_58e-6,
        }
    }

    pub fn acc_to_u(self, a: f32) -> f32 {
        match self.model {
            ThrustModel::Linear => self.hover * a / G,
            _ => ((a.max(0.0) * 2.064 / (4.0 * self.motor_kf())).sqrt() - 150.0) / 850.0,
        }
    }

    pub fn u_to_acc(self, u: f32) -> f32 {
        match self.model {
            ThrustModel::Linear => u / self.hover * G,
            _ => 4.0 * self.motor_kf() * (150.0 + 850.0 * u).powi(2) / 2.064,
        }
    }

    pub fn hover_action(self) -> Action {
        [2.0 / self.tw_ratio - 1.0, 0.0, 0.0, 0.0]
    }

    pub fn acc(self, thrust_action: f32) -> f32 {
        (thrust_action.clamp(-1.0, 1.0) + 1.0) * 0.5 * self.tw_ratio * G
    }

    pub fn action(self, a: Action) -> (Vec3, f32) {
        let [thrust, x, y, z] = a.map(|x| x.clamp(-1.0, 1.0));
        let rates = frd_flu(Vec3::new(x, y, z).component_mul(&self.rate_max));
        (rates, self.acc_to_u(self.acc(thrust)).clamp(0.0, 1.0))
    }

    pub fn level(self, d: Drone) -> (Vec3, f32) {
        let body_z = z(d.r);
        let w = (4.0 * (d.r.transpose() * body_z.cross(&Vec3::z()))).component_div(&self.rate_max);
        let acc = (G - 2.0 * d.v.z).clamp(0.5 * G, 1.8 * G) / body_z.z.max(0.5);
        self.action([2.0 * acc / (self.tw_ratio * G) - 1.0, w.x, w.y, 0.0])
    }
}
