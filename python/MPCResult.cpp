#include "qrw/MPCResult.hpp"

#include "bindings/python.hpp"

void exposeMPCResult() {
  bp::class_<MPCResult>(
      "MPCResult", "MPC result struct.",
      bp::init<uint, uint, uint, uint, bp::optional<uint>>(
          bp::args("self", "Ngait", "nx", "nu", "ndx", "window_size")))
      .def_readwrite("gait", &MPCResult::gait)
      .def_readwrite("xs", &MPCResult::xs)
      .def_readwrite("us", &MPCResult::us)
      .def_readwrite("K", &MPCResult::Ks)
      .def_readwrite("solving_duration", &MPCResult::solving_duration)
      .def_readwrite("num_iters", &MPCResult::num_iters)
      .def_readwrite("new_result", &MPCResult::new_result);
}
