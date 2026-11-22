#ifndef PERCHPILOT_HOLD_PAD_HH_
#define PERCHPILOT_HOLD_PAD_HH_

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gz/math/Pose3.hh>
#include <gz/msgs/double.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/sim/Link.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>

namespace perchpilot
{
class HoldPad final : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemConfigurePriority,
    public gz::sim::ISystemPreUpdate, public gz::sim::ISystemPostUpdate,
    public gz::sim::ISystemReset
{
  public: void Configure(const gz::sim::Entity &,
      const std::shared_ptr<const sdf::Element> &,
      gz::sim::EntityComponentManager &, gz::sim::EventManager &) override;
  public: PriorityType ConfigurePriority() override { return 100; }
  public: void PreUpdate(const gz::sim::UpdateInfo &,
      gz::sim::EntityComponentManager &) override;
  public: void PostUpdate(const gz::sim::UpdateInfo &,
      const gz::sim::EntityComponentManager &) override;
  public: void Reset(const gz::sim::UpdateInfo &,
      gz::sim::EntityComponentManager &) override;

  private: struct Geometry
  {
    gz::math::Pose3d surface;
    gz::math::Pose3d body;
    gz::math::Vector3d normal;
    gz::math::Vector3d padInPlate;
    double contactGap{0};  // distance of the lowest pad/footprint point above the plate surface [m]
    bool supportInside{false};  // the contact point (pad centre or lowest footprint point) lies over the plate
    double supportX{0};
    double supportY{0};
    gz::math::Vector3d plateVelocity;
    gz::math::Vector3d cogVelocity;
    gz::math::Vector3d plateOmega;
    gz::math::Vector3d droneOmega;
    gz::math::Vector3d relativeVelocity;
  };

  private: struct SensorSample
  {
    double time;
    double contact;
    double force;
    double slipU{0};
    double slipW{0};
  };

  private: bool ResolveDrone(const gz::sim::UpdateInfo &,
      gz::sim::EntityComponentManager &);
  private: Geometry ReadGeometry(const gz::sim::EntityComponentManager &) const;
  private: double CollisionNormal(const gz::sim::EntityComponentManager &, const gz::math::Vector3d &normal);
  private: bool Inside(const gz::math::Vector3d &) const;
  private: void Release(const gz::sim::UpdateInfo &, const char *reason);
  private: void OnCommand(const gz::msgs::Double &);
  private: void OnParams(const gz::msgs::Double_V &);

  private: bool configured{false};
  private: std::atomic<bool> holdEnabled{false};
  private: gz::sim::Entity world{gz::sim::kNullEntity};
  private: gz::sim::Link plate;
  private: gz::sim::Link drone;
  private: std::string droneModelName;
  private: std::string droneLinkName;
  private: std::string_view droneLookupError;
  private: std::vector<gz::sim::Entity> droneLinks;
  private: gz::math::Pose3d surfaceInPlate;
  private: gz::math::Vector3d plateHalfSize;
  private: gz::math::Vector3d padInBody;
  private: gz::math::Vector3d cogInBody;
  private: gz::math::Vector3d gravity;
  private: double mass{0};
  private: double mu{0};
  private: double holdForce{0};
  private: std::pair<double, double> sdfParams;
  private: std::mutex paramsMutex;
  private: std::optional<std::pair<double, double>> pendingParams;
  private: double padRadius{0.05};
  private: double padOffset{0.05};
  // Optional landing-gear footprint (half extents in body x/y at the pad plane). 0 = point pad (C2).
  private: double footprintHalfX{0};
  private: double footprintHalfY{0};
  // pad load cell (LoadCell.hh): "total" = servo + drone-plate collision normal force (default), "servo" = old
  private: bool loadCellTotal{true};
  // C5 v2 optical slip sensor (SlipSensor.hh): 2 extra hold_sensors values when enabled (slip_sensor, default 1)
  private: bool slipSensor{true};
  private: double slipScaleErr{0};
  private: double slipGap{1e9};
  private: bool slipPadValid{false};
  private: bool slipPadPrevValid{false};
  private: gz::math::Vector3d slipPad;
  private: gz::math::Vector3d slipPadPrev;
  private: std::uniform_real_distribution<double> uniformScale{-0.1, 0.1};
  private: std::uniform_real_distribution<double> uniform{0.0, 1.0};
  private: double fullScaleMg{2.0};
  private: double collisionNormal{0};
  private: int collisionPoints{0};
  private: bool warnedNoWrench{false};
  private: gz::sim::Entity plateCollision{gz::sim::kNullEntity};
  private: std::set<gz::sim::Entity> droneCollisions;
  private: double engageVnMax{1.5};
  private: double slipGain{0.05};
  private: double noiseFrac{0.03};

  private: bool sampleReady{false};
  private: bool engaged{false};
  private: bool sliding{false};
  private: bool previousContact{false};
  private: double marginN{0};
  private: gz::math::Vector3d anchorInPlate;
  private: gz::math::Quaterniond attitudeInPlate;
  private: gz::math::Vector3d plateVelocityBefore;
  private: gz::math::Vector3d cogVelocityBefore;
  private: gz::math::Vector3d droneOmegaBefore;
  private: gz::math::Vector3d plateAccel;
  private: gz::math::Vector3d measuredThrust;
  private: gz::math::Vector3d appliedForce;

  private: std::chrono::steady_clock::duration nextPublish{
      std::chrono::milliseconds(5)};
  private: std::vector<SensorSample> sensorBuffer;
  private: std::size_t sensorWrite{0};
  private: std::size_t sensorSamples{0};
  private: std::mt19937 random{0x504144};
  private: std::normal_distribution<double> noise{0.0, 1.0};
  private: gz::transport::Node::Publisher sensors;
  private: gz::transport::Node::Publisher truth;
  private: gz::transport::Node::Publisher paramsApplied;
  private: gz::transport::Node::Publisher touchTruth;
  private: gz::transport::Node::Publisher loadTruth;
  private: gz::transport::Node::Publisher slipTruth;
  private: gz::transport::Node node;
};
}
#endif
