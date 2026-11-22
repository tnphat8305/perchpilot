use perch_core::{
    G,
    frames::{
        ActionScale, Mat3, Odometry, ThrustModel, Vec3, ned_frd_enu_flu, quat_to_rot, rot_to_quat,
        rot_y,
    },
    model::{C5_ROWS, C5_V1, C5_V2},
    perception::{Perception, TagPose},
    q2::{Imu, Sensors},
};

fn odometry() -> Odometry {
    Odometry {
        position: [0.0; 3],
        q: [1.0, 0.0, 0.0, 0.0],
        velocity: [1.0, 2.0, 3.0],
        angular_velocity: [1.0, 2.0, 3.0],
        pose_frame: 1,
        velocity_frame: 3,
    }
}

#[test]
fn frames_and_motor_scale() {
    let d = odometry().drone(0.0).unwrap();
    assert_eq!(d.v, Vec3::new(2.0, 1.0, -3.0));
    assert_eq!(d.w, Vec3::new(1.0, -2.0, -3.0));
    let r = rot_y(0.75);
    let px4 = rot_to_quat(ned_frd_enu_flu(r));
    assert!((ned_frd_enu_flu(quat_to_rot(px4)) - r).norm() < 1e-6);

    let sc = ActionScale {
        rate_max: Vec3::new(6.0, 6.0, 3.0),
        tw_ratio: 2.25,
        hover: 0.6,
        model: ThrustModel::X500Perch,
    };
    let (w, thrust) = sc.action([sc.hover_action()[0], 0.25, 0.5, 1.0]);
    assert_eq!(w, Vec3::new(1.5, -3.0, -3.0));
    assert!((thrust - 0.6059).abs() < 5e-4);
    assert!((sc.u_to_acc(thrust) - G).abs() < 1e-5);
}

#[test]
fn stale_and_invalid_measurements() {
    let mut p = Perception::new(0.1);
    assert!(p.drone(0.0).is_none());
    let mut bad = odometry();
    bad.q = [0.0; 4];
    assert!(!p.odometry(bad, 0.0));
    assert!(p.odometry(odometry(), 0.0));
    assert!(p.drone(0.201).is_none());
    let tag = TagPose {
        position: [1.0, 0.0, 0.0],
        q: [1.0, 0.0, 0.0, 0.0],
    };
    assert!(!p.tag(tag, 0.01, -0.01));
    assert!(p.tag(tag, 0.01, 0.01));
    assert!(!p.tag(tag, 0.02, 0.02));
    assert!(
        (p.plate(0.01).unwrap().r.transpose() * p.plate(0.01).unwrap().r - Mat3::identity()).norm()
            < 1e-6
    );
    assert!(!p.plate(0.07).unwrap().visible);
    assert!(!p.plate(2.001).unwrap().valid);
}

#[test]
fn c5_window_consumes_slip_and_resets_on_channel_change() {
    let mut s = Sensors::default();
    s.imu(Imu {
        acc: [1.0, 2.0, 3.0],
        gyro: [4.0, 5.0, 6.0],
    });
    s.hold(&[0.0, 1.0, 8.0], 0.0);
    for i in 0..C5_ROWS {
        let row = s.sample(i as f64 * 0.005, [0.1, 0.0, 1.0]).unwrap();
        assert_eq!(row.len(), C5_V1);
        assert_eq!(&row[..6], &[1.0, -2.0, -3.0, 4.0, -5.0, -6.0]);
    }
    assert_eq!(s.window().unwrap().channels, C5_V1);
    assert_eq!(s.contact(0.4), None);
    for payload in [[0.4, 1.0, 6.0, 0.2, 0.25], [0.4, 1.0, 6.0, 0.3, 0.0]] {
        s.hold(&payload, 0.4);
    }
    let row = s.sample(0.4, [0.0; 3]).unwrap();
    assert_eq!(row.len(), C5_V2);
    assert_eq!(&row[11..], &[0.5, 0.25]);
    assert!(s.window().is_none());
    for i in 1..C5_ROWS {
        assert_eq!(
            &s.sample(0.4 + i as f64 * 0.005, [0.0; 3]).unwrap()[11..],
            &[0.0; 2]
        );
    }
    assert_eq!(s.window().unwrap().slip_mm(), Some(0.5_f32.hypot(0.25)));
}
