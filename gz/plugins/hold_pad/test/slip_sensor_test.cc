// Unit test of the C5 v2 slip-displacement sensor model. Exit code 0 = pass.
#include <cmath>
#include <cstdio>
#include <random>

#include "SlipSensor.hh"

using perchpilot::SlipReading;

static int failures = 0;
#define CHECK(cond, msg)                                              \
  do {                                                                \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; }      \
  } while (0)

int main()
{
  // noiseless: scale error and quantisation (0.01 mm)
  auto r = SlipReading({0.5, -0.25}, 0.0, 0.1, {0.0, 0.0}, 0.5);
  CHECK(std::abs(r[0] - 0.55) < 1e-9 && std::abs(r[1] + 0.28) < 1e-9, "scale error + quantisation");
  r = SlipReading({0.004, 0.0}, 0.0, 0.0, {0.0, 0.0}, 0.5);
  CHECK(r[0] == 0.0, "below one quantum rounds to 0");
  // invalid (pad 5 mm above the surface): true displacement ignored, noise only
  r = SlipReading({1.0, 1.0}, 0.005, 0.0, {1.0, -1.0}, 0.5);
  CHECK(std::abs(r[0] - 0.02) < 1e-9 && std::abs(r[1] + 0.02) < 1e-9, "invalid -> noise only");
  // dropout -> exact zero
  r = SlipReading({1.0, 1.0}, 0.0, 0.0, {3.0, 3.0}, 0.01);
  CHECK(r[0] == 0.0 && r[1] == 0.0, "dropout");
  // statistics: slow slip 0.05 m/s = 0.25 mm per 5 ms sample is resolved over the noise (SNR ~ 12)
  std::mt19937 g(1);
  std::normal_distribution<double> nd(0, 1);
  std::uniform_real_distribution<double> ud(0, 1);
  double sum = 0, sum2 = 0;
  int n = 0;
  for (int i = 0; i < 20000; ++i)
  {
    auto x = SlipReading({0.25, 0.0}, 0.0, 0.0, {nd(g), nd(g)}, ud(g));
    if (x[0] == 0.0 && x[1] == 0.0)
      continue;
    sum += x[0];
    sum2 += x[0] * x[0];
    ++n;
  }
  const double mean = sum / n, sd = std::sqrt(sum2 / n - mean * mean);
  CHECK(std::abs(mean - 0.25) < 0.002 && sd > 0.015 && sd < 0.03, "mean and noise of a slow-slip reading");
  if (failures)
    std::printf("slip_sensor_test: %d failures\n", failures);
  else
    std::printf("slip_sensor_test: all passed\n");
  return failures ? 1 : 0;
}
