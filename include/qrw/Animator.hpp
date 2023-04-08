#pragma once

#include "qrw/Params.hpp"

/// Base high-level motion controller class.
/// Just performs polynomial interpolation.
struct AnimatorBase {
  AnimatorBase(Params& params);
  virtual ~AnimatorBase() = default;

  virtual void update_v_ref(int k, bool gait_is_static);

  inline Eigen::Ref<const Vector6> get_p_ref() const { return p_ref_; }
  inline Eigen::Ref<const Vector6> get_v_ref() const { return v_ref_; }

  //// Data

  Params* params_;

  Vector6 A3_;  // Third order coefficient of the polynomial that generates the
                // velocity profile
  Vector6 A2_;  // Second order coefficient of the polynomial that generates the
                // velocity profile
  Vector6 p_ref_;  // Reference position of the gamepad after low pass filter
  Vector6 v_ref_;  // Reference velocity resulting of the polynomial
                   // interpolation or after low pass filter

  double dt_mpc = 0.0;  // Time step of the MPC
  double dt_wbc = 0.0;  // Time step of the WBC
  int k_mpc = 0;        // Number of WBC time step for one MPC time step

  VectorNi k_switch;
  RowMatrix6N v_switch;

  /// \brief  Handle velocity switch.
  /// \param[in] k index of the current MPC loop.
  void handle_v_switch(int k);
};
