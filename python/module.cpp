#include "bindings/python.hpp"

BOOST_PYTHON_MODULE(quadruped_reactive_walking_pywrap) {
  bp::docstring_options module_docstring_options(true, true, true);
  eigenpy::enableEigenPy();

  bp::import("warnings");

  exposeAnimators();
  exposeParams();
  exposeEstimator();
  exposeFilter();
}
