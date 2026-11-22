// Landing-gear footprint contact geometry for HoldPad (C2 v1.1). Pure C++, no Gazebo types, unit-tested in
// test/footprint_test.cc.
#pragma once

#include <algorithm>
#include <cmath>

namespace perchpilot
{
struct FootprintContact
{
  bool valid{false};   // some footprint point lies over the plate and not deeper than maxPen behind the surface
  double gap{1e9};     // height of the lowest such point above the plate surface [m], clamped at 0
  double x{0.0};       // support point (plate frame) [m]
  double y{0.0};
};

/// Lowest point of the footprint rectangle (half extents hx, hy in the body x/y axes, in the pad plane) that lies
/// over the plate rectangle |x| <= px, |y| <= py. pad: pad centre in the plate frame; R: body -> plate rotation,
/// row-major. Points outside the plate are ignored (a corner over the edge cannot carry the pad); a lowest point
/// more than maxPen behind the surface means the gear passed behind the plate -> invalid.
inline FootprintContact LowestFootprintPoint(const double pad[3], const double R[9], double hx, double hy,
    double px, double py, double maxPen, int n = 21)
{
  FootprintContact best;
  double zmin = 1e9;
  for (int i = 0; i < n; ++i)
  {
    const double s = (n > 1 ? -1.0 + 2.0 * i / (n - 1) : 0.0) * hx;
    for (int j = 0; j < n; ++j)
    {
      const double t = (n > 1 ? -1.0 + 2.0 * j / (n - 1) : 0.0) * hy;
      const double x = pad[0] + R[0] * s + R[1] * t;
      const double y = pad[1] + R[3] * s + R[4] * t;
      const double z = pad[2] + R[6] * s + R[7] * t;
      if (std::abs(x) > px || std::abs(y) > py)
        continue;
      if (z < zmin)
      {
        zmin = z;
        best.x = x;
        best.y = y;
        best.valid = true;
      }
    }
  }
  if (!best.valid || zmin < -maxPen)
  {
    best.valid = false;
    return best;
  }
  best.gap = std::max(zmin, 0.0);
  return best;
}
}  // namespace perchpilot
