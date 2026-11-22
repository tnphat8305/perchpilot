// Pad load cell model for HoldPad (C5 f_normal_meas). Pure C++, unit-tested in test/load_cell_test.cc.
//   mode "total" (default): normal force through the pad/gear = HoldPad servo force + normal component of the
//                           Gazebo collision forces between the drone and the plate (what a load cell under the
//                           pad/gear measures). Noise sigma = noise_frac * FS, saturation +-FS, FS = fs_mg * m * g,
//                           as sim/contact_phase.py. Signed: + compression (plate pushes the drone out), - tension.
//   mode "servo"         : the pre-2026-11-20 reading (servo force only, noise = noise_frac * |reading|), for
//                           reproducing older batches.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace perchpilot
{
/// Sum of the normal components (along the plate's outward normal n) of contact forces acting ON THE DRONE.
inline double CollisionNormalLoad(const std::vector<std::array<double, 3>> &forcesOnDrone,
    const std::array<double, 3> &n)
{
  double sum = 0.0;
  for (const auto &f : forcesOnDrone)
    sum += f[0] * n[0] + f[1] * n[1] + f[2] * n[2];
  return sum;
}

/// Noisy, saturated reading. unitNoise is one N(0, 1) draw.
inline double LoadCellReading(bool totalMode, double servoNormal, double collisionNormal, double noiseFrac,
    double fullScale, double unitNoise)
{
  if (!totalMode)
    return servoNormal + noiseFrac * std::abs(servoNormal) * unitNoise;
  const double f = servoNormal + collisionNormal + noiseFrac * fullScale * unitNoise;
  return std::clamp(f, -fullScale, fullScale);
}
}  // namespace perchpilot
