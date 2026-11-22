#include "HoldMargin.hh"

#include <iomanip>
#include <iostream>
#include <random>

int main()
{
  std::mt19937 random(407);
  std::uniform_real_distribution<double> tiltAngle(0.0, 1.57);
  std::uniform_real_distribution<double> azimuth(-3.141592653589793, 3.141592653589793);
  std::uniform_real_distribution<double> friction(0.0, 1.2);
  std::uniform_real_distribution<double> adhesion(0.0, 3.0 * perchpilot::kGravity);
  std::uniform_real_distribution<double> acceleration(-12.0, 12.0);
  std::uniform_real_distribution<double> thrust(-30.0, 30.0);

  std::cout << "tilt,mu,f_hold,ax,ay,az,nx,ny,nz,tx,ty,tz,margin\n"
            << std::setprecision(17);
  for (int i = 0; i < 1000; ++i)
  {
    const double tilt = tiltAngle(random);
    const double yaw = azimuth(random);
    const double mu = friction(random);
    const double fHold = adhesion(random);
    const perchpilot::Vector plateAccel{
        acceleration(random), acceleration(random), acceleration(random)};
    const perchpilot::Vector normal{-std::sin(tilt) * std::cos(yaw),
        -std::sin(tilt) * std::sin(yaw), std::cos(tilt)};
    const perchpilot::Vector thrustAccel{thrust(random), thrust(random), thrust(random)};
    const auto margin = perchpilot::ComputeHoldMargin(
        mu, fHold, plateAccel, normal, thrustAccel);

    std::cout << tilt << ',' << mu << ',' << fHold;
    for (const auto &vector : {plateAccel, normal, thrustAccel})
      for (const double axis : vector)
        std::cout << ',' << axis;
    std::cout << ',' << margin.Margin() << '\n';
  }
}
