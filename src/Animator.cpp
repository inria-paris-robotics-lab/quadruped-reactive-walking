#include "qrw/Animator.hpp"

AnimatorBase::AnimatorBase(Params &params)
    : params_(&params),
      dt_mpc(params.dt_mpc),
      dt_wbc(params.dt_wbc),
      v_switch(params.v_switch) {
  A2_.setZero();
  A3_.setZero();
  p_ref_.setZero();
  v_ref_.setZero();

  k_mpc = compute_k_mpc(params);
}

void AnimatorBase::update_v_ref(int k, bool) {
  if (k == 0) {
    k_switch = (params_->t_switch / dt_wbc).cast<int>();
  }
  // Polynomial interpolation to generate the velocity profile
  handle_v_switch(k);
}

void AnimatorBase::handle_v_switch(int k) {
  int i = 1;
  while (i < k_switch.size() && k_switch(i) <= k) {
    i++;
  }
  if (i != k_switch.size()) {
    double ev = k - k_switch(i - 1);
    double t1 = k_switch(i) - k_switch(i - 1);
    A3_ = 2 * (v_switch.col(i - 1) - v_switch.col(i)) / pow(t1, 3);
    A2_ = (-3.0 / 2.0) * t1 * A3_;
    v_ref_ = v_switch.col(i - 1) + A2_ * pow(ev, 2) + A3_ * pow(ev, 3);
  }
}
