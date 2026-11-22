// Optical slip-displacement sensor under the pad (docs/CONTRACTS.md C5 v2). Pure C++, unit-tested in
// test/slip_sensor_test.cc.
//   d_meas = (1 + e_s) * d_true + n,  e_s ~ U(-0.1, 0.1) per episode, n ~ N(0, 0.02 mm) per sample,
//   quantised to 0.01 mm, delay 10 ms (HoldPad sensor buffer), valid only with the pad within 2 mm of the
//   surface (else d_true = 0, noise still present), each sample dropped (exact 0) with probability 0.02.
// Units: mm of displacement per 5 ms sample along the pad-plane axes u (lateral) and w (= n x u), not accumulated.
#pragma once

#include <array>
#include <cmath>

namespace perchpilot
{
struct SlipSensorCfg
{
  double noiseMm{0.02};
  double quantMm{0.01};
  double validGapM{0.002};
  double dropout{0.02};
};

/// One reading. dTrueMm: true displacement of the pad relative to the plate over the sample [mm] (u, w);
/// gapM: pad-to-surface distance [m]; scaleErr: e_s of this episode; unitNoise: two N(0,1) draws;
/// uniform: one U(0,1) draw for the dropout.
inline std::array<double, 2> SlipReading(const std::array<double, 2> &dTrueMm, double gapM, double scaleErr,
    const std::array<double, 2> &unitNoise, double uniform, const SlipSensorCfg &c = {})
{
  if (uniform < c.dropout)
    return {0.0, 0.0};
  const bool valid = gapM <= c.validGapM;
  std::array<double, 2> out{};
  for (int i = 0; i < 2; ++i)
  {
    const double d = (valid ? (1.0 + scaleErr) * dTrueMm[i] : 0.0) + c.noiseMm * unitNoise[i];
    out[i] = c.quantMm > 0.0 ? std::round(d / c.quantMm) * c.quantMm : d;
  }
  return out;
}
}  // namespace perchpilot
