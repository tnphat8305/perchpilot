#include "HoldPad.hh"
#include "HoldMargin.hh"
#include "Footprint.hh"
#include "LoadCell.hh"
#include "SlipSensor.hh"

#include <algorithm>
#include <cmath>

#include <gz/common/Console.hh>
#include <gz/msgs/Utility.hh>
#include <gz/msgs/contacts.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/World.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/ExternalWorldWrenchCmd.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Inertial.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Pose.hh>
#include <sdf/Box.hh>

namespace perchpilot
{
namespace sim = gz::sim;
namespace components = gz::sim::components;
using gz::math::Vector3d;

namespace
{
constexpr auto kPublishPeriod = std::chrono::milliseconds(5);
constexpr double kContactDistance = 0.01;
constexpr double kFootprintMaxPenetration = 0.02;  // deeper -> the gear passed behind the plate (review D02)
constexpr double kEngageMaxOutward = 0.05;         // m/s: engage only while approaching (or at rest) (D02)
constexpr double kGripFrequency = 75.0;  // rad/s
constexpr double kAttitudeFrequency = 20.0;  // rad/s

Vector Coordinates(const Vector3d &v)
{
  return {v.X(), v.Y(), v.Z()};
}

Vector3d Tangent(const Vector3d &v, const Vector3d &normal)
{
  return v - v.Dot(normal) * normal;
}
}

void HoldPad::Configure(const sim::Entity &entity,
    const std::shared_ptr<const sdf::Element> &sdf,
    sim::EntityComponentManager &ecm, sim::EventManager &)
{
  droneModelName = sdf->Get<std::string>("drone_model", "").first;
  droneLinkName = sdf->Get<std::string>("drone_link", "").first;
  mu = sdf->Get<double>("mu", -1.0).first;
  holdForce = sdf->Get<double>("f_hold_n", -1.0).first;
  sdfParams = {mu, holdForce};
  padRadius = sdf->Get<double>("pad_radius", padRadius).first;
  padOffset = sdf->Get<double>("pad_offset", padOffset).first;
  footprintHalfX = sdf->Get<double>("footprint_half_x", footprintHalfX).first;
  footprintHalfY = sdf->Get<double>("footprint_half_y", footprintHalfY).first;
  loadCellTotal = sdf->Get<std::string>("load_cell_mode", "total").first != "servo";
  slipSensor = sdf->Get<int>("slip_sensor", 1).first != 0;
  slipScaleErr = uniformScale(random);
  fullScaleMg = sdf->Get<double>("load_cell_fs_mg", fullScaleMg).first;
  engageVnMax = sdf->Get<double>("engage_vn_max", engageVnMax).first;
  slipGain = sdf->Get<double>("slip_gain", slipGain).first;
  noiseFrac = sdf->Get<double>("sensor_noise_frac", noiseFrac).first;
  const double delayMs = sdf->Get<double>("sensor_delay_ms", 10.0).first;
  for (const double value :
      {mu, holdForce, padRadius, padOffset, engageVnMax, slipGain, noiseFrac, delayMs, footprintHalfX,
       footprintHalfY})
  {
    if (!std::isfinite(value) || value < 0.0)
    {
      gzerr << "[HoldPad] Tham số SDF phải hữu hạn và không âm.\n";
      return;
    }
  }

  const sim::Model carrier(entity);
  plate = sim::Link(carrier.LinkByName(ecm, "plate"));
  if (!plate.Valid(ecm))
  {
    gzerr << "[HoldPad] Không tìm thấy link " << carrier.Name(ecm) << "::plate.\n";
    return;
  }
  const auto collision = plate.CollisionByName(ecm, "plate_surface");
  plateCollision = collision;
  const auto *geometry = ecm.Component<components::Geometry>(collision);
  const auto *box = geometry ? geometry->Data().BoxShape() : nullptr;
  if (droneModelName.empty() || droneLinkName.empty() || !box)
  {
    gzerr << "[HoldPad] Cần drone_model, drone_link và collision hộp plate::plate_surface.\n";
    return;
  }
  plateHalfSize = box->Size() * 0.5;
  surfaceInPlate = ecm.ComponentData<components::Pose>(collision).value_or(
      gz::math::Pose3d::Zero);
  surfaceInPlate.Pos() += surfaceInPlate.Rot().RotateVector(
      Vector3d(0, 0, plateHalfSize.Z()));
  plate.EnableVelocityChecks(ecm);

  const double delaySteps = std::ceil(delayMs / kPublishPeriod.count());
  if (delaySteps >= sensorBuffer.max_size())
  {
    gzerr << "[HoldPad] sensor_delay_ms vượt kích thước bộ đệm.\n";
    return;
  }
  sensorBuffer.resize(static_cast<std::size_t>(delaySteps) + 1);
  sensors = node.Advertise<gz::msgs::Double_V>("/perch/hold_sensors");
  truth = node.Advertise<gz::msgs::Double_V>("/perch/hold_truth");
  touchTruth = node.Advertise<gz::msgs::Double_V>("/perch/touch_truth");
  loadTruth = node.Advertise<gz::msgs::Double_V>("/perch/load_truth");
  slipTruth = node.Advertise<gz::msgs::Double_V>("/perch/slip_truth");
  paramsApplied = node.Advertise<gz::msgs::Double_V>("/perch/hold_params_applied");
  configured = sensors && truth && touchTruth && paramsApplied &&
      node.Subscribe("/perch/hold_cmd", &HoldPad::OnCommand, this) &&
      node.Subscribe("/perch/hold_params", &HoldPad::OnParams, this);
  if (!configured)
    gzerr << "[HoldPad] Không khởi tạo được topic Gazebo Transport.\n";
  else
    gzmsg << "[HoldPad] Tìm thấy link " << carrier.Name(ecm) << "::plate (entity "
          << plate.Entity() << ").\n";
}

void HoldPad::OnCommand(const gz::msgs::Double &msg)
{
  holdEnabled.store(msg.data() == 1.0);
}

void HoldPad::OnParams(const gz::msgs::Double_V &msg)
{
  if (msg.data_size() != 2 || !std::isfinite(msg.data(0)) ||
      !std::isfinite(msg.data(1)) || msg.data(0) < 0.0 || msg.data(1) < 0.0)
  {
    gzwarn << "[HoldPad] hold_params cần [mu, f_hold_n] hữu hạn và không âm.\n";
    return;
  }
  const std::lock_guard lock(paramsMutex);
  pendingParams.emplace(msg.data(0), msg.data(1));
}

void HoldPad::Reset(const sim::UpdateInfo &info, sim::EntityComponentManager &)
{
  Release(info, "reset world");
  world = sim::kNullEntity;
  drone = sim::Link();
  droneLinks.clear();
  droneLookupError = {};
  holdEnabled.store(false);
  {
    const std::lock_guard lock(paramsMutex);
    pendingParams.reset();
  }
  mu = sdfParams.first;
  holdForce = sdfParams.second;
  sampleReady = false;
  previousContact = false;
  marginN = 0.0;
  plateAccel = measuredThrust = appliedForce = Vector3d::Zero;
  nextPublish = kPublishPeriod;
  sensorWrite = sensorSamples = 0;
  random.seed(0x504144);
  noise.reset();
}

bool HoldPad::ResolveDrone(const sim::UpdateInfo &info,
    sim::EntityComponentManager &ecm)
{
  if (drone.Valid(ecm))
    return true;

  // Drone có thể được PX4 spawn sau khi world đã chạy.
  Release(info, "mất model/link drone");
  plateAccel = measuredThrust = Vector3d::Zero;
  const auto fail = [this](std::string_view reason)
  {
    if (droneLookupError != reason)
    {
      gzerr << "[HoldPad] " << droneModelName << "::" << droneLinkName << ": "
            << reason << " (world entity " << world << "), sẽ tìm lại.\n";
      droneLookupError = reason;
    }
    return false;
  };
  // Configure có thể chạy trước khi Gazebo đặt world làm cha của model mang tấm.
  if (world == sim::kNullEntity)
  {
    world = sim::worldEntity(plate.Entity(), ecm);
    if (world == sim::kNullEntity)
      return fail("không tìm thấy world chứa tấm bám");
    gravity = sim::World(world).Gravity(ecm).value_or(Vector3d(0, 0, -kGravity));
  }

  const sim::Model model(sim::World(world).ModelByName(ecm, droneModelName));
  if (!model.Valid(ecm))
    return fail("không tìm thấy model");
  const auto link = model.LinkByName(ecm, droneLinkName);
  if (link == sim::kNullEntity)
    return fail("không tìm thấy link");
  const auto *inertial = ecm.Component<components::Inertial>(link);
  if (!inertial)
    return fail("link thiếu Inertial");

  mass = 0.0;
  droneLinks.clear();
  droneCollisions.clear();
  for (const auto entity : ecm.Descendants(model.Entity()))
  {
    if (!ecm.Component<components::Link>(entity))
      continue;
    droneLinks.push_back(entity);
    for (const auto child : ecm.ChildrenByComponents(entity, components::Collision()))
      droneCollisions.insert(child);
    if (const auto *component = ecm.Component<components::Inertial>(entity))
      mass += component->Data().MassMatrix().Mass();
  }
  if (mass <= 0.0)
    return fail("tổng khối lượng model không dương");
  cogInBody = inertial->Data().Pose().Pos();
  padInBody = cogInBody - Vector3d(0, 0, padOffset);
  drone = sim::Link(link);
  drone.EnableVelocityChecks(ecm);
  droneLookupError = {};
  gzmsg << "[HoldPad] Tìm thấy model " << droneModelName << " (entity "
        << model.Entity() << "), link " << droneLinkName << " (entity " << link
        << "), khối lượng " << mass << " kg.\n";
  return true;
}

double HoldPad::CollisionNormal(const sim::EntityComponentManager &ecm, const Vector3d &normal)
{
  collisionPoints = 0;
  const auto *data = ecm.Component<components::ContactSensorData>(plateCollision);
  if (!data)
    return 0.0;
  std::vector<std::array<double, 3>> forces;
  bool contactWithoutWrench = false;
  for (const auto &c : data->Data().contact())
  {
    const auto e1 = static_cast<sim::Entity>(c.collision1().id());
    const auto e2 = static_cast<sim::Entity>(c.collision2().id());
    const bool plateFirst = e1 == plateCollision && droneCollisions.count(e2);
    const bool plateSecond = e2 == plateCollision && droneCollisions.count(e1);
    if (!plateFirst && !plateSecond)
      continue;
    if (c.wrench_size() == 0)
      contactWithoutWrench = true;
    for (const auto &w : c.wrench())
    {
      // Sign convention measured in SITL (q2_probe_m3, 2026-11-20): with the plate as the listed body, its
      // wrench projects POSITIVE on the outward normal at rest (about +5 N on a gear resting on the 65 deg plate),
      // i.e. it is the force pushing the drone out, so it is used as the force on the drone.
      const auto &fp = plateFirst ? w.body_1_wrench().force() : w.body_2_wrench().force();
      forces.push_back({fp.x(), fp.y(), fp.z()});
      ++collisionPoints;
    }
  }
  if (contactWithoutWrench && !warnedNoWrench)
  {
    gzwarn << "[HoldPad] drone-plate contacts carry no wrench data: load cell = servo force only.\n";
    warnedNoWrench = true;
  }
  return CollisionNormalLoad(forces, {normal.X(), normal.Y(), normal.Z()});
}

HoldPad::Geometry HoldPad::ReadGeometry(const sim::EntityComponentManager &ecm) const
{
  Geometry result;
  const auto platePose = sim::worldPose(plate.Entity(), ecm);
  result.surface = platePose * surfaceInPlate;
  result.body = sim::worldPose(drone.Entity(), ecm);
  result.normal = result.surface.Rot().RotateVector(Vector3d::UnitZ);
  const auto pad = result.body.Pos() + result.body.Rot().RotateVector(padInBody);
  result.padInPlate = result.surface.Rot().RotateVectorReverse(
      pad - result.surface.Pos());
  // Point pad: |gap| of the pad point. Footprint: the lowest corner of the landing-gear rectangle, so a gear
  // that meets the plate a few degrees off-flat still engages (a skid otherwise hits before the pad point).
  result.contactGap = std::abs(result.padInPlate.Z());
  result.supportInside = std::abs(result.padInPlate.X()) <= plateHalfSize.X() &&
      std::abs(result.padInPlate.Y()) <= plateHalfSize.Y();
  if (footprintHalfX > 0.0 || footprintHalfY > 0.0)
  {
    // C2 v1.1 + review D02: lowest footprint point that lies OVER the plate, limited penetration
    const auto Rbp = result.surface.Rot().Inverse() * result.body.Rot();
    const gz::math::Matrix3d M(Rbp);
    const double R[9] = {M(0, 0), M(0, 1), M(0, 2), M(1, 0), M(1, 1), M(1, 2), M(2, 0), M(2, 1), M(2, 2)};
    const double pad[3] = {result.padInPlate.X(), result.padInPlate.Y(), result.padInPlate.Z()};
    const auto fp = LowestFootprintPoint(pad, R, footprintHalfX, footprintHalfY, plateHalfSize.X(),
        plateHalfSize.Y(), kFootprintMaxPenetration);
    result.contactGap = fp.valid ? fp.gap : 1e9;
    result.supportInside = fp.valid;
    result.supportX = fp.x;
    result.supportY = fp.y;
  }
  result.plateVelocity = plate.WorldLinearVelocity(ecm).value_or(Vector3d::Zero);
  result.cogVelocity = drone.WorldLinearVelocity(ecm, cogInBody).value_or(
      Vector3d::Zero);
  result.plateOmega = plate.WorldAngularVelocity(ecm).value_or(Vector3d::Zero);
  result.droneOmega = drone.WorldAngularVelocity(ecm).value_or(Vector3d::Zero);
  const auto padVelocity = result.cogVelocity +
      result.droneOmega.Cross(result.body.Rot().RotateVector(padInBody - cogInBody));
  const auto surfaceVelocity = result.plateVelocity +
      result.plateOmega.Cross(pad - platePose.Pos());
  result.relativeVelocity = padVelocity - surfaceVelocity;
  return result;
}

bool HoldPad::Inside(const Vector3d &pad) const
{
  return std::abs(pad.X()) <= plateHalfSize.X() &&
      std::abs(pad.Y()) <= plateHalfSize.Y();
}

void HoldPad::Release(const sim::UpdateInfo &info, const char *reason)
{
  if (engaged)
    gzmsg << "[HoldPad] Nhả " << droneModelName << "::" << droneLinkName << " ở t="
          << std::chrono::duration<double>(info.simTime).count() << " s, " << reason << ".\n";
  engaged = sliding = false;
}

void HoldPad::PreUpdate(const sim::UpdateInfo &info, sim::EntityComponentManager &ecm)
{
  if (info.paused || info.dt <= info.dt.zero() || !configured)
    return;
  std::optional<std::pair<double, double>> params;
  {
    const std::lock_guard lock(paramsMutex);
    params.swap(pendingParams);
  }
  if (params)
  {
    mu = params->first;
    holdForce = params->second;
    gz::msgs::Double_V msg;
    msg.add_data(mu);
    msg.add_data(holdForce);
    paramsApplied.Publish(msg);
  }
  appliedForce = Vector3d::Zero;
  marginN = 0.0;
  sampleReady = ResolveDrone(info, ecm);
  if (!sampleReady)
    return;

  const auto geometry = ReadGeometry(ecm);
  plateVelocityBefore = geometry.plateVelocity;
  cogVelocityBefore = geometry.cogVelocity;
  droneOmegaBefore = geometry.droneOmega;
  Vector3d externalForce;
  bool hasExternalForce = false;
  // Đọc tổng lực của cả model, kể cả rotor, trước khi ghi lực HoldPad.
  for (const auto link : droneLinks)
  {
    const auto *wrench = ecm.Component<components::ExternalWorldWrenchCmd>(link);
    if (wrench && wrench->Data().has_force())
    {
      externalForce += gz::msgs::Convert(wrench->Data().force());
      hasExternalForce = true;
    }
  }
  const auto thrustAccel = hasExternalForce ? externalForce / mass : measuredThrust;
  // C2 dùng g = 9,81 cố định, như hàm Python.
  const auto loads = ComputeHoldMargin(mu, holdForce / mass,
      Coordinates(plateAccel), Coordinates(geometry.normal), Coordinates(thrustAccel));
  marginN = mass * loads.Margin();

  const double normalVelocity = geometry.relativeVelocity.Dot(geometry.normal);
  const bool inside = Inside(geometry.padInPlate);
  const bool enabled = holdEnabled.load();
  const bool wasEngaged = engaged;
  const bool normalOk = loads.normalOk >= 0.0;
  if (!enabled || !inside || !normalOk)
  {
    const char *reason = !enabled ? "lệnh hold_cmd=0" :
        !inside ? "qua mép tấm" : "vượt lực giữ pháp tuyến";
    Release(info, reason);
    return;
  }
  const bool canEngage = geometry.contactGap <= kContactDistance && geometry.supportInside &&
      -normalVelocity <= engageVnMax && normalVelocity <= kEngageMaxOutward;
  if (!wasEngaged && !canEngage)
    return;
  engaged = true;

  const bool wasSliding = sliding;
  sliding = loads.frictionOk < 0.0;
  // Neo XY và góc ở thời điểm gắn, hoặc đặt lại neo XY khi hết trượt.
  if (!wasEngaged)
  {
    attitudeInPlate = geometry.surface.Rot().Inverse() * geometry.body.Rot();
    slipScaleErr = uniformScale(random);        // C5 v2: e_s fixed per episode (= per touchdown)
    gzmsg << "[HoldPad] Gắn " << droneModelName << "::" << droneLinkName << " ở t="
          << std::chrono::duration<double>(info.simTime).count() << " s, khoảng cách="
          << geometry.padInPlate.Z() << " m, v_n=" << normalVelocity
          << " m/s, margin=" << marginN << " N.\n";
  }
  if (!wasEngaged || wasSliding)
    anchorInPlate.Set(geometry.padInPlate.X(), geometry.padInPlate.Y(), 0);

  const auto requiredAccel = plateAccel - gravity - thrustAccel;
  const auto requiredTangent = Tangent(requiredAccel, geometry.normal);
  const double frequency = std::min(kGripFrequency,
      0.35 / std::chrono::duration<double>(info.dt).count());
  const double spring = frequency * frequency;
  const double damping = 2.0 * frequency;
  // Footprint mode (C2 v1.1): the gear, not the pad centre, rests on the plate, so the normal spring acts on the
  // lowest footprint point. Using the pad-centre height (3-6 cm with a 15-20 deg misaligned gear) saturated the
  // grip at -f_hold (permanent tension) and made the load cell read -F_hold regardless of thrust.
  const bool footprint = footprintHalfX > 0.0 || footprintHalfY > 0.0;
  const double gap = footprint && geometry.supportInside ? geometry.contactGap : geometry.padInPlate.Z();
  const double normalForce = std::max(-holdForce, mass *
      (requiredAccel.Dot(geometry.normal) - spring * gap -
      damping * normalVelocity));
  const auto tangentVelocity = Tangent(geometry.relativeVelocity, geometry.normal);
  Vector3d tangentAccel;
  if (sliding)
  {
    const auto slipDirection = -requiredTangent.Normalized();
    const auto targetVelocity = slipGain * (-marginN) * slipDirection;
    tangentAccel = requiredTangent + damping * (targetVelocity - tangentVelocity);
  }
  else
  {
    const auto offset = geometry.surface.Rot().RotateVector(Vector3d(
        geometry.padInPlate.X() - anchorInPlate.X(),
        geometry.padInPlate.Y() - anchorInPlate.Y(), 0));
    tangentAccel = requiredTangent - spring * offset - damping * tangentVelocity;
  }

  const auto targetAttitude = geometry.surface.Rot() * attitudeInPlate;
  auto attitudeError = targetAttitude * geometry.body.Rot().Inverse();
  attitudeError.Normalize();
  Vector3d axis;
  double angle;
  attitudeError.AxisAngle(axis, angle);
  angle = std::remainder(angle, 2.0 * GZ_PI);
  const auto angularAccel = kAttitudeFrequency * kAttitudeFrequency * angle * axis +
      2.0 * kAttitudeFrequency * (geometry.plateOmega - geometry.droneOmega);
  auto torque = drone.WorldInertiaMatrix(ecm).value_or(gz::math::Matrix3d::Zero) *
      angularAccel;
  const double maxTorque = padRadius * (std::max(normalForce, 0.0) + holdForce);
  if (torque.Length() > maxTorque)
    torque = torque.Normalized() * maxTorque;

  appliedForce = normalForce * geometry.normal + mass * tangentAccel;
  drone.AddWorldForce(ecm, appliedForce);
  drone.AddWorldWrench(ecm, Vector3d::Zero, torque);
}

void HoldPad::PostUpdate(const sim::UpdateInfo &info,
    const sim::EntityComponentManager &ecm)
{
  if (info.paused || info.dt <= info.dt.zero() || !configured)
    return;

  bool contact = false;
  double normalForce = 0.0;
  double slipSpeed = 0.0;
  if (sampleReady && drone.Valid(ecm))
  {
    const auto geometry = ReadGeometry(ecm);
    const double dt = std::chrono::duration<double>(info.dt).count();
    plateAccel = (geometry.plateVelocity - plateVelocityBefore) / dt;
    measuredThrust = (geometry.cogVelocity - cogVelocityBefore) / dt -
        gravity - appliedForce / mass;
    const bool inside = Inside(geometry.padInPlate);
    if (!inside)
      Release(info, "qua mép tấm");
    contact = inside &&
        (engaged || (geometry.contactGap <= kContactDistance && geometry.supportInside));
    if (contact && !previousContact)
    {
      // Review D01: score at the PAD point like the GPU sim (v_CoG + w x r_pad), relative to the plate surface
      // point under it, with the velocities from before the impact step; the CoG values are published too.
      const auto rPad = geometry.body.Rot().RotateVector(padInBody - cogInBody);
      const auto padWorld = geometry.body.Pos() + geometry.body.Rot().RotateVector(padInBody);
      const auto platePos = sim::worldPose(plate.Entity(), ecm).Pos();
      const auto velocity = cogVelocityBefore + droneOmegaBefore.Cross(rPad) -
          (plateVelocityBefore + geometry.plateOmega.Cross(padWorld - platePos));
      const auto velocityCog = cogVelocityBefore - plateVelocityBefore;
      const auto up = geometry.body.Rot().RotateVector(Vector3d::UnitZ);
      gz::msgs::Double_V touch;
      touch.add_data(std::chrono::duration<double>(info.simTime).count());
      touch.add_data(geometry.padInPlate.X());
      touch.add_data(geometry.padInPlate.Y());
      touch.add_data(geometry.padInPlate.Z());
      touch.add_data(std::acos(std::clamp(up.Dot(geometry.normal), -1.0, 1.0)) * 180.0 / GZ_PI);
      touch.add_data(-velocity.Dot(geometry.normal));
      touch.add_data(Tangent(velocity, geometry.normal).Length());
      touch.add_data(droneOmegaBefore.Length());
      touch.add_data(-velocityCog.Dot(geometry.normal));
      touch.add_data(Tangent(velocityCog, geometry.normal).Length());
      touch.add_data(geometry.supportX);
      touch.add_data(geometry.supportY);
      touchTruth.Publish(touch);
    }
    normalForce = contact ? appliedForce.Dot(geometry.normal) : 0.0;
    slipPad = geometry.padInPlate;
    slipGap = engaged ? 0.0 : geometry.contactGap;
    slipPadValid = true;
    collisionNormal = CollisionNormal(ecm, geometry.normal);
    if (sliding)
      slipSpeed = Tangent(geometry.relativeVelocity, geometry.normal).Length();
  }
  previousContact = contact;

  if (info.simTime < nextPublish)
    return;
  nextPublish = info.simTime - info.simTime % kPublishPeriod + kPublishPeriod;
  const double time = std::chrono::duration<double>(info.simTime).count();
  gz::msgs::Double_V msg;
  msg.add_data(time);
  msg.add_data(engaged);
  msg.add_data(marginN);
  msg.add_data(slipSpeed);
  truth.Publish(msg);

  // C5 v2 slip displacement over this 5 ms sample, pad-plane axes u = surface y (lateral), w = n x u = -surface x
  std::array<double, 2> dTrue{0.0, 0.0};
  if (slipPadValid && slipPadPrevValid)
    dTrue = {1000.0 * (slipPad.Y() - slipPadPrev.Y()), -1000.0 * (slipPad.X() - slipPadPrev.X())};
  slipPadPrev = slipPad;
  slipPadPrevValid = slipPadValid;
  const auto slipMeas = SlipReading(dTrue, slipGap, slipScaleErr, {noise(random), noise(random)},
      uniform(random));
  gz::msgs::Double_V st;                 // truth for checking: [t, d_u mm, d_w mm, valid (gap <= 2 mm)]
  st.add_data(time);
  st.add_data(dTrue[0]);
  st.add_data(dTrue[1]);
  st.add_data(slipGap <= SlipSensorCfg{}.validGapM ? 1.0 : 0.0);
  slipTruth.Publish(st);
  sensorBuffer[sensorWrite] = {time, contact ? 1.0 : 0.0,
      LoadCellReading(loadCellTotal, normalForce, collisionNormal, noiseFrac,
          fullScaleMg * mass * 9.81, noise(random)), slipMeas[0], slipMeas[1]};
  gz::msgs::Double_V load;              // diagnostics: [t, servo N, collision N, contact points with wrench]
  load.add_data(time);
  load.add_data(normalForce);
  load.add_data(collisionNormal);
  load.add_data(collisionPoints);
  loadTruth.Publish(load);
  sensorWrite = (sensorWrite + 1) % sensorBuffer.size();
  sensorSamples = std::min(sensorSamples + 1, sensorBuffer.size());
  if (sensorSamples == sensorBuffer.size())
  {
    const auto &sample = sensorBuffer[sensorWrite];
    msg.clear_data();
    msg.add_data(sample.time);
    msg.add_data(sample.contact);
    msg.add_data(sample.force);
    if (slipSensor)
    {
      msg.add_data(sample.slipU);
      msg.add_data(sample.slipW);
    }
    sensors.Publish(msg);
  }
}
}

GZ_ADD_PLUGIN(perchpilot::HoldPad, gz::sim::System,
    gz::sim::ISystemConfigure, gz::sim::ISystemConfigurePriority,
    gz::sim::ISystemPreUpdate, gz::sim::ISystemPostUpdate, gz::sim::ISystemReset)
