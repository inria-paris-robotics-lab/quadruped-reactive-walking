#include "bindings/python.hpp"

BOOST_PYTHON_MODULE(quadruped_reactive_walking) {
  exposeJoystick();
  exposeParams();
}