#pragma once

#include "qrw/Params.hpp"
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/multibody/data.hpp>

namespace qrw {

using Motion = pinocchio::MotionTpl<Scalar>;

class IOCPAbstract {
 public:
  using ConstVecRefN = Eigen::Ref<const VectorN>;
  IOCPAbstract(Params const& params);
  virtual ~IOCPAbstract() = default;

  virtual void make_ocp(uint k, const ConstVecRefN& x0, Matrix34 footsteps,
                        Motion base_vel_ref) = 0;
  virtual void solve(std::size_t k) = 0;

  Params params_;
  uint num_iters_;
};

}  // namespace qrw
