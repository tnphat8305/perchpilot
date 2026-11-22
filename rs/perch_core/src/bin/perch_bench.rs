use perch_core::{
    frames::{ActionScale, Drone, Mat3, Plate, ThrustModel, Vec3, rot_y},
    model::{Belief, C5_ROWS, C5_V2, Policy},
    phase::{Input, Machine, Phase, PhaseConfig},
};
use std::{env, hint::black_box, io, path::Path, time::Instant};

fn measure(name: &str, mut f: impl FnMut(usize)) {
    for i in 0..100 {
        f(i);
    }
    let mut samples: Vec<_> = (0..5000)
        .map(|i| {
            let start = Instant::now();
            f(i);
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_unstable_by(f64::total_cmp);
    println!(
        "{name:12} p50 {:.5} ms  p95 {:.5} ms  p99 {:.5} ms",
        samples[2499], samples[4749], samples[4949]
    );
}

fn main() -> io::Result<()> {
    let root = env::args_os()
        .nth(1)
        .ok_or_else(|| io::Error::other("usage: perch_bench FIXTURE_DIR"))?;
    let root = Path::new(&root);
    let mut policy = Policy::load(&root.join("policy"))?;
    let mut belief = Belief::load(&root.join("belief"))?;
    let mut obs: Vec<f32> = (0..policy.meta.obs_dim)
        .map(|i| (i as f32 / 10.0).sin())
        .collect();
    let mut window = [[0.0; C5_V2]; C5_ROWS];
    for (i, x) in window.iter_mut().enumerate() {
        *x = [
            0.03 * (i as f32).sin(),
            0.05,
            9.81,
            0.0,
            0.01,
            -0.01,
            8.0,
            1.0,
            0.1,
            1.0,
            0.1,
            0.01,
            0.0,
        ];
    }
    measure("policy", |i| {
        obs[0] = -((i % 500) as f32) / 500.0;
        black_box(policy.infer(black_box(&obs)));
    });
    measure("belief", |i| {
        window[0][0] = (i % 500) as f32 / 1000.0;
        black_box(belief.infer(black_box(&window)));
    });

    let scale = ActionScale {
        rate_max: policy.meta.rate_max.into(),
        tw_ratio: policy.meta.tw_ratio_nominal,
        hover: 0.6,
        model: ThrustModel::X500Perch,
    };
    let cfg = PhaseConfig {
        max_steps: usize::MAX,
        ..Default::default()
    };
    let mut machine = Machine::new(cfg, policy, scale, 0.227, Phase::Perch, None);
    let plate = Plate {
        c: Vec3::new(6.0, 0.0, 1.4),
        r: rot_y(-1.1345),
        v: Vec3::new(0.05, 0.0, 0.0),
        valid: true,
        age: 0.0,
        visible_for: 1.0,
        reliability: 1.0,
        visible: true,
        loss_s: 0.0,
    };
    measure("PERCH tick", |i| {
        let t = i as f64 * 0.01 + 1.0;
        let drone = Drone {
            t,
            p: Vec3::new(2.0, 0.03, 2.5),
            v: Vec3::zeros(),
            r: Mat3::identity(),
            w: Vec3::zeros(),
        };
        black_box(
            machine
                .step(black_box(Input {
                    t,
                    drone: Some(drone),
                    armed: true,
                    offboard: true,
                    plate: Some(plate),
                    contact: Some(false),
                    rates_mode: Some(true),
                    window: None,
                }))
                .unwrap(),
        );
    });
    Ok(())
}
