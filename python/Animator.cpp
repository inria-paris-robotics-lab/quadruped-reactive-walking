#include "qrw/Animator.hpp"
#ifdef QRW_JOYSTICK_SUPPORT
#include "qrw/Joystick.hpp"
#endif

#include "bindings/python.hpp"

template <typename Class>
struct AnimatorVisitor : public bp::def_visitor<AnimatorVisitor<Class>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(
          bp::init<Params&>(bp::args("self", "params"), "Default constructor."))

        .def("update_v_ref", &Class::update_v_ref,
             bp::args("self", "k", "gait_is_static"), "Update joystick values.")
        .def("handle_v_switch", &Class::handle_v_switch, bp::args("self", "k"),
             "Handle velocity switch.\n"
             ":param k: index of the current MPC cycle.")
        .def("get_p_ref", &Class::get_p_ref, bp::args("self"),
             "Get Reference Position")
        .def("get_v_ref", &Class::get_v_ref, bp::args("self"),
             "Get Reference Velocity");
  }
};

void exposeAnimators() {
  bp::class_<AnimatorBase>(
      "AnimatorBase",
      "Base animator class. Handles animation using polynomial interpolation.",
      bp::no_init)
      .def(AnimatorVisitor<AnimatorBase>());

#ifdef QRW_JOYSTICK_SUPPORT
  bp::class_<Joystick>("Joystick",
                       "Animator using an external joystick peripheral.",
                       bp::no_init)
      .def(AnimatorVisitor<Joystick>())
      .def("get_joystick_code", &Joystick::getJoystickCode, bp::args("self"),
           "Get Joystick Code")
      .def("get_start", &Joystick::getStart, bp::args("self"),
           "Get Joystick Start")
      .def("get_stop", &Joystick::getStop, bp::args("self"),
           "Get Joystick Stop")
      .def("get_cross", &Joystick::getCross, bp::args("self"),
           "Get Joystick Cross status")
      .def("get_circle", &Joystick::getCircle, bp::args("self"),
           "Get Joystick Circle status")
      .def("get_triangle", &Joystick::getTriangle, bp::args("self"),
           "Get Joystick Triangle status")
      .def("get_square", &Joystick::getSquare, bp::args("self"),
           "Get Joystick Square status")
      .def("get_l1", &Joystick::getL1, bp::args("self"),
           "Get Joystick L1 status")
      .def("get_r1", &Joystick::getR1, bp::args("self"),
           "Get Joystick R1 status");
#endif
}
