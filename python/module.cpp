#include "bindings/python.hpp"
#include "qrw/Types.h"

BOOST_PYTHON_MODULE(quadruped_reactive_walking_pywrap) {
  bp::docstring_options module_docstring_options(true, true, true);
  eigenpy::enableEigenPy();

  bp::import("warnings");

  eigenpy::enableEigenPySpecific<Vector6>();

  exposeAnimators();
  exposeParams();
  exposeEstimator();
  exposeFilter();
}
