#ifndef __PYTHON_ADDER__
#define __PYTHON_ADDER__

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/python.hpp>
#include <boost/python/scope.hpp>

#undef BOOST_BIND_GLOBAL_PLACEHOLDERS

#include <eigenpy/eigenpy.hpp>

namespace bp = boost::python;

void exposeJoystick();
void exposeParams();
void exposeEstimator();
void exposeFilter();

#endif
