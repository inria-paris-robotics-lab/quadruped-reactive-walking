#include "qrw/Estimator.hpp"

#include <example-robot-data/path.hpp>

#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/math/rpy.hpp"
#include "pinocchio/parsers/urdf.hpp"

Estimator::Estimator()
    : perfectEstimator_(false),
      solo3D_(false),
      dt_(0.0),
      initialized_(false),
      feetFrames_(Vector4i::Zero()),
      footRadius_(0.0155),
      alphaPos_({0.995, 0.995, 0.9}),
      alphaVelMax_(1.),
      alphaVelMin_(0.97),
      alphaSecurity_(0.),
      IMUYawOffset_(0.),
      IMULinearAcceleration_(Vector3::Zero()),
      IMUAngularVelocity_(Vector3::Zero()),
      IMURpy_(Vector3::Zero()),
      IMUQuat_(pinocchio::SE3::Quaternion(1.0, 0.0, 0.0, 0.0)),
      qActuators_(Vector12::Zero()),
      vActuators_(Vector12::Zero()),
      phaseRemainingDuration_(0),
      feetStancePhaseDuration_(Vector4::Zero()),
      feetStatus_(Vector4::Zero()),
      feetTargets_(Matrix34::Zero()),
      q_FK_(Vector19::Zero()),
      v_FK_(Vector18::Zero()),
      baseVelocityFK_(Vector3::Zero()),
      basePositionFK_(Vector3::Zero()),
      b_baseVelocity_(Vector3::Zero()),
      feetPositionBarycenter_(Vector3::Zero()),
      qEstimate_(Vector19::Zero()),
      vEstimate_(Vector18::Zero()),
      vSecurity_(Vector12::Zero()),
      windowSize_(0),
      vFiltered_(Vector6::Zero()),
      qRef_(Vector18::Zero()),
      vRef_(Vector18::Zero()),
      baseVelRef_(Vector6::Zero()),
      baseAccRef_(Vector6::Zero()),
      oRh_(Matrix3::Identity()),
      hRb_(Matrix3::Identity()),
      oTh_(Vector3::Zero()),
      h_v_(Vector6::Zero()),
      h_vFiltered_(Vector6::Zero()) {
  b_M_IMU_ = pinocchio::SE3(pinocchio::SE3::Quaternion(1.0, 0.0, 0.0, 0.0),
                            Vector3(0.1163, 0.0, 0.02));
  q_FK_(6) = 1.0;
  qEstimate_(6) = 1.0;
}

void Estimator::initialize(Params &params) {
  dt_ = params.dt_wbc;
  perfectEstimator_ = params.perfect_estimator;
  solo3D_ = params.solo3D;

  // Filtering estimated linear velocity
  int k_mpc = compute_k_mpc(params);
  windowSize_ = (uint)(k_mpc * params.gait.rows() / params.N_periods);
  vx_queue_.resize(windowSize_, 0.0);  // List full of 0.0
  vy_queue_.resize(windowSize_, 0.0);  // List full of 0.0
  vz_queue_.resize(windowSize_, 0.0);  // List full of 0.0

  // Filtering velocities used for security checks
  double fc = 6.0;
  double y = 1 - std::cos(2 * M_PI * fc * dt_);
  alphaSecurity_ = -y + std::sqrt(y * y + 2 * y);

  // Initialize Quantities
  basePositionFK_(2) = params.h_ref;
  velocityFilter_.initialize(dt_, Vector3::Zero(), Vector3::Zero());
  positionFilter_.initialize(dt_, Vector3::Zero(), basePositionFK_);
  qRef_(2, 0) = params.h_ref;
  qRef_.tail(12) = Vector12(params.q_init.data());

  // Initialize Pinocchio
  const std::string filename = std::string(
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf");
  pinocchio::urdf::buildModel(filename, pinocchio::JointModelFreeFlyer(),
                              model_, false);
  data_ = pinocchio::Data(model_);
  pinocchio::computeAllTerms(model_, data_, qEstimate_, vEstimate_);
  feetFrames_ << (int)model_.getFrameId("FL_FOOT"),
      (int)model_.getFrameId("FR_FOOT"), (int)model_.getFrameId("HL_FOOT"),
      (int)model_.getFrameId("HR_FOOT");
}

void Estimator::run(MatrixN const &gait, MatrixN const &feetTargets,
                    VectorN const &baseLinearAcceleration,
                    VectorN const &baseAngularVelocity,
                    VectorN const &baseOrientation, VectorN const &q,
                    VectorN const &v, VectorN const &perfectPosition,
                    Vector3 const &b_perfectVelocity) {
  updateFeetStatus(gait, feetTargets);
  updateIMUData(baseLinearAcceleration, baseAngularVelocity, baseOrientation,
                perfectPosition);
  updateJointData(q, v);

  updateForwardKinematics();
  computeFeetPositionBarycenter();

  estimateVelocity(b_perfectVelocity);
  estimatePosition(perfectPosition.head(3));

  filterVelocity();

  vSecurity_ = (1 - alphaSecurity_) * vActuators_ + alphaSecurity_ * vSecurity_;
}

void Estimator::updateReferenceState(VectorN const &vRef) {
  // Update reference acceleration and velocities
  Matrix3 Rz = pinocchio::rpy::rpyToMatrix(0., 0., -baseVelRef_[5] * dt_);
  baseAccRef_.head(3) = (vRef.head(3) - Rz * baseVelRef_.head(3)) / dt_;
  baseAccRef_.tail(3) = (vRef.tail(3) - Rz * baseVelRef_.tail(3)) / dt_;
  baseVelRef_ = vRef;

  // Update position and velocity state vectors
  Rz = pinocchio::rpy::rpyToMatrix(0., 0., qRef_[5]);
  vRef_.head(2) = Rz.topLeftCorner(2, 2) * baseVelRef_.head(2);
  vRef_[5] = baseVelRef_[5];
  vRef_.tail(12) = vActuators_;

  qRef_.head(2) += vRef_.head(2) * dt_;
  qRef_[2] = qEstimate_[2];
  qRef_.segment(3, 2) = IMURpy_.head(2);
  qRef_[5] += baseVelRef_[5] * dt_;
  qRef_.tail(12) = qActuators_;

  // Transformation matrices
  hRb_ = pinocchio::rpy::rpyToMatrix(IMURpy_[0], IMURpy_[1], 0.);
  oRh_ = pinocchio::rpy::rpyToMatrix(0., 0., qRef_[5]);
  oTh_.head(2) = qRef_.head(2);

  // Express estimated velocity and filtered estimated velocity in horizontal
  // frame
  h_v_.head(3) = hRb_ * vEstimate_.head(3);
  h_v_.tail(3) = hRb_ * vEstimate_.segment(3, 3);
  h_vFiltered_.head(3) = hRb_ * vFiltered_.head(3);
  h_vFiltered_.tail(3) = hRb_ * vFiltered_.tail(3);
}

void Estimator::updateFeetStatus(MatrixN const &gait,
                                 MatrixN const &feetTargets) {
  feetStatus_ = gait.row(0);
  feetTargets_ = feetTargets;

  feetStancePhaseDuration_ += feetStatus_;
  feetStancePhaseDuration_ = feetStancePhaseDuration_.cwiseProduct(feetStatus_);

  phaseRemainingDuration_ = 1;
  while (feetStatus_.isApprox((Vector4)gait.row(phaseRemainingDuration_))) {
    phaseRemainingDuration_++;
    if (phaseRemainingDuration_ >= gait.rows()) break;
  }
}

void Estimator::updateIMUData(Vector3 const &baseLinearAcceleration,
                              Vector3 const &baseAngularVelocity,
                              Vector3 const &baseOrientation,
                              VectorN const &perfectPosition) {
  IMULinearAcceleration_ = baseLinearAcceleration;
  IMUAngularVelocity_ = baseAngularVelocity;
  IMURpy_ = baseOrientation;

  if (!initialized_) {
    IMUYawOffset_ = IMURpy_(2);
    initialized_ = true;
  }
  IMURpy_(2) -= IMUYawOffset_;

  if (solo3D_) {
    IMURpy_.tail(1) = perfectPosition.tail(1);
  }

  IMUQuat_ = pinocchio::rpy::rpyToMatrix(IMURpy_(0), IMURpy_(1), IMURpy_(2));
}

void Estimator::updateJointData(Vector12 const &q, Vector12 const &v) {
  qActuators_ = q;
  vActuators_ = v;
}

void Estimator::updateForwardKinematics() {
  q_FK_.tail(12) = qActuators_;
  v_FK_.tail(12) = vActuators_;
  q_FK_.segment(3, 4) << 0., 0., 0., 1.;
  pinocchio::forwardKinematics(model_, data_, q_FK_, v_FK_);

  q_FK_.segment(3, 4) = IMUQuat_.coeffs();

  int nContactFeet = 0;
  Vector3 baseVelocityEstimate = Vector3::Zero();
  Vector3 basePositionEstimate = Vector3::Zero();
  for (int foot = 0; foot < 4; foot++) {
    if (feetStatus_(foot) == 1. && feetStancePhaseDuration_[foot] >= 16) {
      baseVelocityEstimate += computeBaseVelocityFromFoot(foot);
      basePositionEstimate += computeBasePositionFromFoot(foot);
      nContactFeet++;
    }
  }

  if (nContactFeet > 0) {
    baseVelocityFK_ = baseVelocityEstimate / nContactFeet;
    basePositionFK_ = basePositionEstimate / nContactFeet;
  }
}

Vector3 Estimator::computeBaseVelocityFromFoot(int footId) {
  pinocchio::updateFramePlacement(model_, data_, (uint)feetFrames_[footId]);
  pinocchio::SE3 contactFrame = data_.oMf[(uint)feetFrames_[footId]];
  Vector3 frameVelocity =
      pinocchio::getFrameVelocity(model_, data_, (uint)feetFrames_[footId],
                                  pinocchio::LOCAL)
          .linear();
  frameVelocity(0) +=
      footRadius_ * (vActuators_(1 + 3 * footId) + vActuators_(2 + 3 * footId));
  return contactFrame.translation().cross(IMUAngularVelocity_) -
         contactFrame.rotation() * frameVelocity;
}

Vector3 Estimator::computeBasePositionFromFoot(int footId) {
  pinocchio::updateFramePlacement(model_, data_, (uint)feetFrames_[footId]);
  Vector3 basePosition = -data_.oMf[(uint)feetFrames_[footId]].translation();

  return basePosition;
}

void Estimator::computeFeetPositionBarycenter() {
  int nContactFeet = 0;
  Vector3 feetPositions = Vector3::Zero();
  for (int j = 0; j < 4; j++) {
    if (feetStatus_(j) == 1.) {
      feetPositions += feetTargets_.col(j);
      nContactFeet++;
    }
  }
  if (nContactFeet > 0) feetPositionBarycenter_ = feetPositions / nContactFeet;
}

double Estimator::computeAlphaVelocity() {
  double a = std::ceil(feetStancePhaseDuration_.maxCoeff() * 0.1) - 1;
  double b = static_cast<double>(phaseRemainingDuration_);
  double c = ((a + b) - 2) * 0.5;
  if (a <= 0 || b <= 1)
    return alphaVelMax_;
  else
    return alphaVelMin_ +
           (alphaVelMax_ - alphaVelMin_) * std::abs(c - (a - 1)) / c;
}

void Estimator::estimateVelocity(Vector3 const &b_perfectVelocity) {
  Vector3 alpha = Vector3::Ones() * computeAlphaVelocity();
  Matrix3 oRb = IMUQuat_.toRotationMatrix();
  Vector3 bTi = (b_M_IMU_.translation()).cross(IMUAngularVelocity_);

  // At IMU location in world frame
  Vector3 oi_baseVelocityFK =
      solo3D_ ? oRb * (b_perfectVelocity + bTi) : oRb * (baseVelocityFK_ + bTi);
  Vector3 oi_baseVelocity = velocityFilter_.compute(
      oi_baseVelocityFK, oRb * IMULinearAcceleration_, alpha);

  // At base location in base frame
  b_baseVelocity_ = oRb.transpose() * oi_baseVelocity - bTi;

  vEstimate_.head(3) = perfectEstimator_ ? b_perfectVelocity : b_baseVelocity_;
  vEstimate_.segment(3, 3) = IMUAngularVelocity_;
  vEstimate_.tail(12) = vActuators_;
}

void Estimator::estimatePosition(Vector3 const &perfectPosition) {
  Matrix3 oRb = IMUQuat_.toRotationMatrix();

  Vector3 basePosition =
      solo3D_ ? perfectPosition : (basePositionFK_ + feetPositionBarycenter_);
  qEstimate_.head(3) =
      positionFilter_.compute(basePosition, oRb * b_baseVelocity_, alphaPos_);

  if (perfectEstimator_ || solo3D_) qEstimate_(2) = perfectPosition(2);
  qEstimate_.segment(3, 4) = IMUQuat_.coeffs();
  qEstimate_.tail(12) = qActuators_;
}

void Estimator::filterVelocity() {
  vFiltered_ = vEstimate_.head(6);
  vx_queue_.pop_back();
  vy_queue_.pop_back();
  vz_queue_.pop_back();
  vx_queue_.push_front(vEstimate_(0));
  vy_queue_.push_front(vEstimate_(1));
  vz_queue_.push_front(vEstimate_(2));
  vFiltered_(0) =
      std::accumulate(vx_queue_.begin(), vx_queue_.end(), 0.) / windowSize_;
  vFiltered_(1) =
      std::accumulate(vy_queue_.begin(), vy_queue_.end(), 0.) / windowSize_;
  vFiltered_(2) =
      std::accumulate(vz_queue_.begin(), vz_queue_.end(), 0.) / windowSize_;
}
