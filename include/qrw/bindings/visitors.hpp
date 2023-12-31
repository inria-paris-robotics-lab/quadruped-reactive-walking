#include "qrw/bindings/python.hpp"

#include "qrw/Animator.hpp"

namespace qrw {

template <typename Class>
struct AnimatorVisitor : public bp::def_visitor<AnimatorVisitor<Class>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(
          bp::init<Params&>(bp::args("self", "params"), "Default constructor."))

        .def("update_v_ref", &Class::update_v_ref,
             bp::args("self", "k", "gait_is_static"), "Update joystick values.")
        .def("get_p_ref", &Class::get_p_ref, bp::args("self"),
             "Get Reference Position")
        .def("get_v_ref", &Class::get_v_ref, bp::args("self"),
             "Get Reference Velocity");
  }
};

}  // namespace qrw
