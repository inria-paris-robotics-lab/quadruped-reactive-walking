#pragma once

#include "qrw/Params.hpp"

/// \brief Base class for high-level inputs, i.e. the base velocity.
struct AnimatorBase {
  AnimatorBase(Params const& params);
  virtual ~AnimatorBase() = default;

  /// \brief  Handle velocity switch.
  /// \param[in] k index of the current MPC loop.
  void handle_v_switch(int k);

  /// \brief Update the desired base velocity.
  /// \param[in] k Numero of the current loop
  virtual void update_v_ref(int k, bool gait_is_static);

  //// Data

  Params const* params_;

  /// 3rd order coefficient of the velocity profile polynomial
  Vector6 A3_;
  /// 2nd order coefficient of the velocity profile polynomial
  Vector6 A2_;
  /// Desired reference position.
  Vector6 p_ref_;
  /// Desired reference velocity.
  Vector6 v_ref_;

  /// Time step of the MPC
  double dt_mpc = 0.0;
  /// Time step of the whole-body controller
  double dt_wbc = 0.0;
  /// Number of WBC time steps for one MPC time step
  int k_mpc = 0;

  /// Time for switches in the predefined velocity
  VectorN t_switch;
  /// Indexes for switches
  VectorNi k_switch;
  /// Velocity values at the switching times
  RowMatrix6N v_switch;
};
