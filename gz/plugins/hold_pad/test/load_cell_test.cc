// Unit test of the pad load cell model. Exit code 0 = pass.
#include <cmath>
#include <cstdio>

#include "LoadCell.hh"

using perchpilot::CollisionNormalLoad;
using perchpilot::LoadCellReading;

static int failures = 0;
#define CHECK(cond, msg)                                              \
  do {                                                                \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; }      \
  } while (0)

int main()
{
  const double s = std::sin(65.0 * M_PI / 180.0), c = std::cos(65.0 * M_PI / 180.0);
  const std::array<double, 3> n{-s, 0.0, c};                // plate normal, backwards-up (perch_world)
  // two skid contacts pushing the drone out along n with 3 N and 2 N, plus a tangential friction part
  const std::array<double, 3> t{c, 0.0, s};
  std::vector<std::array<double, 3>> f{{3 * n[0] + 1 * t[0], 3 * n[1], 3 * n[2] + 1 * t[2]},
                                       {2 * n[0], 2 * n[1], 2 * n[2]}};
  CHECK(std::abs(CollisionNormalLoad(f, n) - 5.0) < 1e-12, "normal part of the collision forces");
  CHECK(CollisionNormalLoad({}, n) == 0.0, "no contacts");
  const double fs = 2.0 * 2.0643 * 9.81;
  // total mode: servo + collision, noise 3% FS, signed, saturated
  CHECK(std::abs(LoadCellReading(true, 1.5, 5.0, 0.03, fs, 0.0) - 6.5) < 1e-12, "total = servo + collision");
  CHECK(std::abs(LoadCellReading(true, -8.0, 0.0, 0.03, fs, 1.0) - (-8.0 + 0.03 * fs)) < 1e-12, "noise is 3% FS");
  CHECK(LoadCellReading(true, 100.0, 0.0, 0.03, fs, 0.0) == fs, "saturates at +FS");
  CHECK(LoadCellReading(true, -100.0, 0.0, 0.03, fs, 0.0) == -fs, "saturates at -FS");
  // servo mode: old behaviour (collision ignored, noise proportional to the reading)
  CHECK(std::abs(LoadCellReading(false, 2.0, 5.0, 0.03, fs, 1.0) - 2.06) < 1e-12, "servo mode");
  if (failures)
    std::printf("load_cell_test: %d failures\n", failures);
  else
    std::printf("load_cell_test: all passed\n");
  return failures ? 1 : 0;
}
