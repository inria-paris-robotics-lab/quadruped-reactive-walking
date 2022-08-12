#include "bindings/python.hpp"

BOOST_PYTHON_MODULE(quadruped_reactive_walking_pywrap) {
  exposeJoystick();
  exposeParams();
  exposeEstimator();
  exposeFilter();
}