///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Filter class
///
/// \details This class applies a low pass filter to estimated data to avoid
/// keeping high frequency components into
///          what is given to the "low frequency" model predictive control
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef FILTER_H_INCLUDED
#define FILTER_H_INCLUDED

#include <deque>

#include "qrw/Params.hpp"

class LowPassFilter {
 public:
  /// \brief Constructor
  LowPassFilter(Params const& params);

  /// \brief Run one iteration of the filter and return the filtered measurement
  ///
  /// \param[in] x Quantity to filter
  /// \param[in] check_modulo Check for the +-pi modulo of orientation if true
  ConstVecRefN filter(Vector6 const& x, bool check_modulo);

  /// \brief Add or remove 2 PI to all elements in the queues to handle +- pi
  /// modulo
  ///
  /// \param[in] a Angle that needs change (3, 4, 5 for Roll, Pitch, Yaw
  /// respectively) \param[in] dir Direction of the change (+pi to -pi or -pi to
  /// +pi)
  void handle_modulo(int a, bool dir);

  ConstVecRefN getFilt() const { return y_; }

 private:
  double b_;       // Denominator coefficients of the filter transfer function
  Vector2 a_;      // Numerator coefficients of the filter transfer function
  Vector6 x_;      // Latest measurement
  VectorN y_;      // Latest result
  Vector6 accum_;  // Used to compute the result (accumulation)

  std::deque<Vector6> x_queue_;  // Store the last measurements for product with
                                 // denominator coefficients
  std::deque<Vector6> y_queue_;  // Store the last results for product with
                                 // numerator coefficients

  bool init_;  // Initialisation flag
};

#endif  // FILTER_H_INCLUDED
