#include "qrw/IOCPAbstract.hpp"

#include "bindings/python.hpp"
#include <eigenpy/optional.hpp>

namespace qrw {

struct OCPWrapper : IOCPAbstract, bp::wrapper<IOCPAbstract> {
  using IOCPAbstract::IOCPAbstract;

  void make_ocp(uint k, const ConstVecRefN &x0, Matrix34 footsteps,
                Motion base_vel_ref) override {
    bp::override fn = get_override("make_ocp");
    fn(k, x0, footsteps, base_vel_ref);
  }

  void solve(std::size_t k) override {
    bp::override fn = get_override("solve");
    fn(k);
  }

  bp::tuple get_results(boost::optional<uint> window_size = boost::none) {
    return get_override("get_results")(window_size);
  }
};

void exposeSolverInterface() {
  eigenpy::OptionalConverter<uint>::registration();
  bp::class_<OCPWrapper, boost::noncopyable>("IOCPAbstract",
                                             "Base OCP interface.", bp::no_init)
      .def(bp::init<Params const &>(bp::args("self", "params")))
      .def("solve", bp::pure_virtual(&OCPWrapper::solve), bp::args("self", "k"))
      .def("make_ocp", bp::pure_virtual(&OCPWrapper::make_ocp),
           bp::args("self", "k"))
      .def("get_results", bp::pure_virtual(&OCPWrapper::get_results),
           (bp::arg("self"), bp::arg("window_size") = boost::none),
           "Fetch the results of the latest MPC iteration.")
      .def_readonly("params", &IOCPAbstract::params_)
      .def_readwrite("num_iters", &IOCPAbstract::num_iters_);
}

}  // namespace qrw
