#include "qrw/bindings/python.hpp"
#include "qrw/bindings/yaml-node.hpp"
#include "qrw/Types.h"
#include <eigenpy/std-vector.hpp>

BOOST_PYTHON_MODULE(quadruped_reactive_walking_pywrap) {
  bp::docstring_options module_docstring_options(true, true, true);
  eigenpy::enableEigenPy();

  bp::import("warnings");
  bp::import("pinocchio");
  bp::import("crocoddyl");

  eigenpy::enableEigenPySpecific<Vector6>();
  eigenpy::enableEigenPySpecific<RowMatrix6N>();
  using StdVecVectorN = std::vector<VectorN>;
  using StdVecMatrixN = std::vector<MatrixN>;
  eigenpy::StdVectorPythonVisitor<StdVecVectorN, true>::expose("StdVecVectorN");
  eigenpy::StdVectorPythonVisitor<StdVecMatrixN, true>::expose("StdVecMatrixN");

  qrw::YamlNodeToPython::registration();

  qrw::exposeAnimators();
  qrw::exposeParams();
  qrw::exposeEstimator();
  qrw::exposeFilter();
  qrw::exposeMPCResult();
  qrw::exposeSolverInterface();
  qrw::exposeMPCInterface();
  qrw::exposeResidualFlyHigh();
}
