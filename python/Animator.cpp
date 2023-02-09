#include "qrw/Joystick.hpp"

#include "bindings/python.hpp"

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

void exposeJoystick() {
  bp::class_<AnimatorBase>("AnimatorBase", bp::no_init)
      .def(AnimatorVisitor<AnimatorBase>());

  bp::class_<Joystick>("Joystick", bp::no_init)
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
}
