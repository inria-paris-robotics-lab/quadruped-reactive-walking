#ifndef __PYTHON_ADDER__
#define __PYTHON_ADDER__

#include <eigenpy/eigenpy.hpp>

namespace bp = boost::python;

namespace qrw {
void exposeAnimators();
void exposeParams();
void exposeEstimator();
void exposeFilter();
void exposeMPCResult();
void exposeSolverInterface();
void exposeMPCInterface();
void exposeResidualFlyHigh();
}  // namespace qrw

#endif
