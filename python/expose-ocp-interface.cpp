#include "qrw/IOCPAbstract.hpp"

#include "bindings/python.hpp"

namespace qrw {

struct OCPWrapper : IOCPAbstract, bp::wrapper<IOCPAbstract> {
  using IOCPAbstract::IOCPAbstract;

  void solve(std::size_t k) override {
    if (bp::override fn = get_override("solve")) {
      fn(k);
    }
    PyErr_SetString(PyExc_NotImplementedError, "Method not implemented.");
    bp::throw_error_already_set();
  }
};

void exposeSolverInterface() {
  bp::class_<OCPWrapper, boost::noncopyable>("IOCPAbstract",
                                             "Base OCP interface.", bp::no_init)
      .def(bp::init<Params const &>(bp::args("self", "params")))
      .def("solve", bp::pure_virtual(&OCPWrapper::solve), bp::args("self", "k"))
      .def_readonly("params", &IOCPAbstract::params_);
}

}  // namespace qrw
