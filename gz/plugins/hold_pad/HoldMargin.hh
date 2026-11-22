#ifndef PERCHPILOT_HOLD_MARGIN_HH_
#define PERCHPILOT_HOLD_MARGIN_HH_

#include <algorithm>
#include <array>
#include <cmath>

namespace perchpilot
{
using Vector = std::array<double, 3>;
inline constexpr double kGravity = 9.81;

struct HoldMargin
{
  double fn;
  double normalOk;
  double frictionOk;

  double Margin() const { return std::min(normalOk, frictionOk); }
};

inline HoldMargin ComputeHoldMargin(double mu, double fHold,
    const Vector &plateAccel, const Vector &normal, const Vector &thrustAccel)
{
  const Vector force{plateAccel[0] - thrustAccel[0],
      plateAccel[1] - thrustAccel[1],
      plateAccel[2] + kGravity - thrustAccel[2]};
  const double fn = force[0] * normal[0] + force[1] * normal[1] +
      force[2] * normal[2];
  const double ft = std::hypot(force[0] - fn * normal[0],
      force[1] - fn * normal[1], force[2] - fn * normal[2]);
  return {fn, fn + fHold, mu * (std::max(fn, 0.0) + fHold) - ft};
}
}
#endif
