// Unit test of the footprint contact geometry (D02 review cases). Exit code 0 = pass.
#include <cmath>
#include <cstdio>

#include "Footprint.hh"

using perchpilot::LowestFootprintPoint;

static int failures = 0;
#define CHECK(cond, msg)                                     \
  do {                                                       \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } \
  } while (0)

static void RotY(double a, double R[9])
{
  const double c = std::cos(a), s = std::sin(a);
  const double M[9] = {c, 0, s, 0, 1, 0, -s, 0, c};
  for (int i = 0; i < 9; ++i)
    R[i] = M[i];
}

int main()
{
  const double hx = 0.125, hy = 0.132, px = 0.225, py = 0.225, pen = 0.02;
  double R[9];
  // 1. flat gear 5 mm above the plate centre -> valid, gap 5 mm
  RotY(0.0, R);
  double pad1[3] = {0.0, 0.0, 0.005};
  auto c = LowestFootprintPoint(pad1, R, hx, hy, px, py, pen);
  CHECK(c.valid && std::abs(c.gap - 0.005) < 1e-9, "flat gear gap");
  // 2. review D02 case: pad centre (x, z) = (0.205, 0.055), gear tilted 25 deg about y: the lowest corner hangs
  //    over the edge (x > 0.225); the lowest point OVER the plate is still ~4.6 cm away -> no engage at 1 cm
  RotY(25.0 * M_PI / 180.0, R);
  double pad2[3] = {0.205, 0.0, 0.055};
  c = LowestFootprintPoint(pad2, R, hx, hy, px, py, pen, 101);
  CHECK(c.valid && c.gap > 0.04 && c.gap < 0.05, "corner over the edge ignored");
  CHECK(c.x <= px + 1e-12, "support point inside the plate");
  // 3. gear 3 cm behind the surface (passed through) -> invalid
  RotY(0.0, R);
  double pad3[3] = {0.0, 0.0, -0.03};
  c = LowestFootprintPoint(pad3, R, hx, hy, px, py, pen);
  CHECK(!c.valid, "behind the plate");
  // 4. gear entirely beside the plate -> invalid
  double pad4[3] = {0.6, 0.0, 0.0};
  c = LowestFootprintPoint(pad4, R, hx, hy, px, py, pen);
  CHECK(!c.valid, "outside the plate");
  // 5. tilted 20 deg with the low corner over the plate -> gap from that corner (0 when touching)
  RotY(20.0 * M_PI / 180.0, R);                 // +x end goes down
  double pad5[3] = {0.0, 0.0, 0.125 * std::sin(20.0 * M_PI / 180.0)};
  c = LowestFootprintPoint(pad5, R, hx, hy, px, py, pen);
  CHECK(c.valid && c.gap < 1e-6, "one skid end on the plate");
  if (failures)
    std::printf("footprint_test: %d failures\n", failures);
  else
    std::printf("footprint_test: all passed\n");
  return failures ? 1 : 0;
}
