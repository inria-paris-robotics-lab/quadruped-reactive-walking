#ifndef __PYTHON_ADDER__
#define __PYTHON_ADDER__

#include <eigenpy/eigenpy.hpp>

namespace bp = boost::python;

namespace qrw {
void exposeSolverInterface();
void exposeMPCInterface();
}  // namespace qrw

void exposeAnimators();
void exposeParams();
void exposeEstimator();
void exposeFilter();
void exposeMPCResult();

#endif
