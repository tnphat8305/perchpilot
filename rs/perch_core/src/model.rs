use crate::{Action, invalid};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::BTreeMap, fs, io, path::Path};

pub const C5_ROWS: usize = 80;
pub const C5_V1: usize = 11;
pub const C5_V2: usize = 13;
pub type C5Row = [f32; C5_V2];
pub type C5Samples = [C5Row; C5_ROWS];

#[derive(Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    offset: usize,
}

#[derive(Deserialize)]
struct Header {
    version: u32,
    kind: String,
    file: String,
    tensors: BTreeMap<String, Tensor>,
    metadata: Value,
}

struct Weights {
    header: Header,
    bin: Vec<u8>,
}

impl Weights {
    fn load(run: &Path, kind: &str) -> io::Result<Self> {
        let path = run.join(format!("{kind}.rust.json"));
        let header: Header = serde_json::from_slice(&fs::read(&path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("{}: {e}; run ros/export_weights.py", path.display()),
            )
        })?)?;
        if header.version != 1
            || header.kind != kind
            || header.file.contains(['/', '\\'])
            || !header.file.ends_with(".bin")
        {
            return Err(invalid("invalid Rust weights header"));
        }
        let bin = fs::read(run.join(&header.file))?;
        Ok(Self { header, bin })
    }

    fn shape(&self, name: &str) -> io::Result<&[usize]> {
        self.header
            .tensors
            .get(name)
            .map(|t| t.shape.as_slice())
            .ok_or_else(|| invalid(format!("missing tensor {name}")))
    }

    fn tensor(&self, name: &str, shape: &[usize]) -> io::Result<Vec<f32>> {
        if self.shape(name)? != shape || shape.contains(&0) {
            return Err(invalid(format!("{name}: expected shape {shape:?}")));
        }
        let offset = self.header.tensors[name].offset;
        let data = shape
            .iter()
            .try_fold(4_usize, |bytes, dim| bytes.checked_mul(*dim))
            .and_then(|bytes| offset.checked_add(bytes))
            .and_then(|end| self.bin.get(offset..end))
            .ok_or_else(|| invalid(format!("{name}: tensor outside file")))?;
        let values: Vec<_> = data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(format!("{name}: non-finite weights")));
        }
        Ok(values)
    }
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Activation {
    Elu,
    Tanh,
    Relu,
    Linear,
}

struct Linear {
    weights: Vec<f32>,
    bias: Vec<f32>,
    activation: Activation,
}

impl Linear {
    fn load(
        w: &Weights,
        name: &str,
        input: usize,
        output: usize,
        activation: Activation,
    ) -> io::Result<Self> {
        Ok(Self {
            weights: w.tensor(&format!("{name}.weight"), &[output, input])?,
            bias: w.tensor(&format!("{name}.bias"), &[output])?,
            activation,
        })
    }

    fn run(&self, input: &[f32], output: &mut [f32]) {
        for ((dst, row), bias) in output
            .iter_mut()
            .zip(self.weights.chunks_exact(input.len()))
            .zip(&self.bias)
        {
            // Independent partial sums allow LLVM to vectorize the dot product on f32.
            let mut accum = [0.0; 8];
            let (chunks, remainder) = row.as_chunks::<8>();
            let (inputs, tail) = input.as_chunks::<8>();
            for (w, x) in chunks.iter().zip(inputs) {
                for i in 0..8 {
                    accum[i] += w[i] * x[i];
                }
            }
            let val = accum.iter().sum::<f32>()
                + remainder.iter().zip(tail).map(|(w, x)| w * x).sum::<f32>()
                + bias;
            *dst = match self.activation {
                Activation::Elu if val < 0.0 => val.exp_m1(),
                Activation::Tanh => val.tanh(),
                Activation::Relu => val.max(0.0),
                _ => val,
            };
        }
    }
}

#[derive(Deserialize)]
struct Layer {
    name: String,
    activation: Activation,
}

const OBS_V1: [(&str, usize); 9] = [
    ("rel_p_uwn", 3),
    ("rel_v_uwn", 3),
    ("R_body_in_plate", 9),
    ("omega_body", 3),
    ("up_in_plate", 3),
    ("prev_act", 4),
    ("t_norm", 1),
    ("veh_speed", 1),
    ("plate_tilt", 1),
];
const TAG_OBS: [(&str, usize); 3] = [("rho", 1), ("visible", 1), ("t_loss_norm", 1)];

#[derive(Clone, Debug, Deserialize)]
pub struct PolicyMeta {
    pub obs_dim: usize,
    pub obs_layout: Vec<(String, usize)>,
    pub act_layout: [String; 4],
    pub rate_max: [f32; 3],
    pub tw_ratio_nominal: f32,
    pub dt: f64,
    pub frame: String,
}

impl PolicyMeta {
    fn validate(&self) -> io::Result<()> {
        if self.obs_dim < 28
            || self.obs_layout.iter().map(|(_, n)| n).sum::<usize>() != self.obs_dim
            || !self
                .obs_layout
                .iter()
                .take(OBS_V1.len())
                .map(|(s, n)| (s.as_str(), *n))
                .eq(OBS_V1)
            || self.act_layout != ["thrust01", "wx", "wy", "wz"]
            || self.frame != "plate_uwn"
            || (self.dt - 0.01).abs() > 1e-9
            || !(self.obs_dim - 28 - usize::from(self.has_tag()) * 3).is_multiple_of(4)
            || self.rate_max.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || !self.tw_ratio_nominal.is_finite()
            || self.tw_ratio_nominal <= 0.0
        {
            return Err(invalid("policy metadata violates C1"));
        }
        Ok(())
    }

    pub fn has_tag(&self) -> bool {
        self.obs_layout
            .iter()
            .skip(self.obs_layout.len().saturating_sub(3))
            .map(|(s, n)| (s.as_str(), *n))
            .eq(TAG_OBS)
    }

    pub fn history_len(&self) -> usize {
        (self.obs_dim - 28 - usize::from(self.has_tag()) * 3) / 4
    }
}

pub struct Policy {
    pub meta: PolicyMeta,
    mean: Vec<f32>,
    denominator: Vec<f32>,
    layers: Vec<Linear>,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Policy {
    pub fn load(run: &Path) -> io::Result<Self> {
        let w = Weights::load(run, "policy")?;
        let meta: PolicyMeta = match fs::read(run.join("policy.json")) {
            Ok(data) => serde_json::from_slice(&data)?,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                serde_json::from_value(w.header.metadata["policy"].clone())?
            }
            Err(err) => return Err(err),
        };
        meta.validate()?;
        let mean = w.tensor("norm.mean", &[meta.obs_dim])?;
        let mut denominator = w.tensor("norm.var", &[meta.obs_dim])?;
        for x in &mut denominator {
            if *x < 0.0 {
                return Err(invalid("policy variance must be nonnegative"));
            }
            *x = x.sqrt() + 1e-6;
        }
        let spec: Vec<Layer> = serde_json::from_value(w.header.metadata["layers"].clone())?;
        let mut dim = meta.obs_dim;
        let mut max_dim = dim;
        let mut layers = Vec::new();
        for layer in spec {
            let shape = w.shape(&format!("{}.weight", layer.name))?;
            let Some(&out) = shape.first() else {
                return Err(invalid("missing layer dimension"));
            };
            layers.push(Linear::load(&w, &layer.name, dim, out, layer.activation)?);
            dim = out;
            max_dim = max_dim.max(dim);
        }
        if dim != 4 || layers.is_empty() {
            return Err(invalid("policy must output four actions"));
        }
        Ok(Self {
            meta,
            mean,
            denominator,
            layers,
            a: vec![0.0; max_dim],
            b: vec![0.0; max_dim],
        })
    }

    pub fn infer(&mut self, obs: &[f32]) -> Action {
        assert_eq!(obs.len(), self.meta.obs_dim);
        for (((dst, x), m), d) in self
            .a
            .iter_mut()
            .zip(obs)
            .zip(&self.mean)
            .zip(&self.denominator)
        {
            *dst = ((x - m) / d).clamp(-10.0, 10.0);
        }
        let mut n = obs.len();
        for l in &self.layers {
            l.run(&self.a[..n], &mut self.b);
            n = l.bias.len();
            std::mem::swap(&mut self.a, &mut self.b);
        }
        std::array::from_fn(|i| self.a[i].clamp(-1.0, 1.0))
    }
}

struct Conv {
    weights: Vec<f32>,
    bias: Vec<f32>,
    time: usize,
    stride: usize,
}

impl Conv {
    fn load(
        w: &Weights,
        name: &str,
        input: usize,
        output: usize,
        time: usize,
        stride: usize,
    ) -> io::Result<Self> {
        Ok(Self {
            weights: w.tensor(&format!("{name}.weight"), &[output, input, 5])?,
            bias: w.tensor(&format!("{name}.bias"), &[output])?,
            time,
            stride,
        })
    }

    fn run(&self, input: &[f32], patches: &mut [f32], output: &mut [f32]) {
        let n = self.time / self.stride;
        let patch_len = input.len() / self.time * 5 * n;
        let patches = &mut patches[..patch_len];
        // Pack each tap in time order. The inner multiply-add then uses contiguous slices.
        for (series, taps) in input
            .chunks_exact(self.time)
            .zip(patches.chunks_exact_mut(5 * n))
        {
            for (tap, patch) in taps.chunks_exact_mut(n).enumerate() {
                for (t, dst) in patch.iter_mut().enumerate() {
                    let signed = t as isize * self.stride as isize + tap as isize - 2;
                    *dst = series.get(signed as usize).copied().unwrap_or(0.0);
                }
            }
        }
        for ((row, w), bias) in output
            .chunks_exact_mut(n)
            .zip(self.weights.chunks_exact(patch_len / n))
            .zip(&self.bias)
        {
            row.fill(*bias);
            for (patch, weight) in patches.chunks_exact(n).zip(w) {
                for (accum, x) in row.iter_mut().zip(patch) {
                    *accum += x * weight;
                }
            }
            for x in row {
                *x = x.max(0.0);
            }
        }
    }
}

pub struct Belief {
    pub channels: usize,
    mean: Vec<f32>,
    std: Vec<f32>,
    mask: Vec<f32>,
    temperature: f32,
    conv: [Conv; 3],
    hidden: Linear,
    output: Linear,
    a: Vec<f32>,
    b: Vec<f32>,
    patches: Vec<f32>,
}

impl Belief {
    pub fn load(run: &Path) -> io::Result<Self> {
        let w = Weights::load(run, "belief")?;
        let &[channels] = w.shape("mean")? else {
            return Err(invalid("expected belief mean vector"));
        };
        let &[ch, _, _] = w.shape("conv.0.weight")? else {
            return Err(invalid("expected Conv1d weights"));
        };
        let temperature = w.tensor("temp", &[1])?[0];
        let std = w.tensor("std", &[channels])?;
        let mask = w.tensor("mask", &[channels])?;
        if !matches!(channels, C5_V1 | C5_V2)
            || ch == 0
            || temperature <= 0.0
            || std.iter().any(|&s| s <= 0.0)
        {
            return Err(invalid("invalid C5 belief normalizer or channel count"));
        }
        let max_ch = ch.max(channels);
        Ok(Self {
            channels,
            mean: w.tensor("mean", &[channels])?,
            std,
            mask,
            temperature,
            conv: [
                Conv::load(&w, "conv.0", channels, ch, C5_ROWS, 1)?,
                Conv::load(&w, "conv.2", ch, ch, C5_ROWS, 2)?,
                Conv::load(&w, "conv.4", ch, ch, C5_ROWS / 2, 2)?,
            ],
            hidden: Linear::load(&w, "head.1", ch * (C5_ROWS / 4), 128, Activation::Relu)?,
            output: Linear::load(&w, "head.3", 128, 3, Activation::Linear)?,
            a: vec![0.0; (max_ch * C5_ROWS).max(128)],
            b: vec![0.0; (max_ch * C5_ROWS).max(128)],
            patches: vec![0.0; max_ch * 5 * C5_ROWS],
        })
    }

    pub fn infer(&mut self, window: &C5Samples) -> [f32; 3] {
        for (c, series) in self
            .a
            .chunks_exact_mut(C5_ROWS)
            .take(self.channels)
            .enumerate()
        {
            for (dst, row) in series.iter_mut().zip(window) {
                *dst = (row[c] - self.mean[c]) / self.std[c] * self.mask[c];
            }
        }
        let mut n = self.channels * C5_ROWS;
        for l in &self.conv {
            l.run(&self.a[..n], &mut self.patches, &mut self.b);
            n = l.time / l.stride * l.bias.len();
            std::mem::swap(&mut self.a, &mut self.b);
        }
        self.hidden.run(&self.a[..n], &mut self.b);
        let mut o = [0.0; 3];
        self.output.run(&self.b[..128], &mut o);
        o[0] = 1.0 / (1.0 + (-o[0] / self.temperature).exp());
        o[1] = 1.0 / (1.0 + (-o[1]).exp());
        o
    }
}
