#include "qrw/Joystick.hpp"

#include "bindings/python.hpp"

template <typename Joystick>
struct JoystickVisitor : public bp::def_visitor<JoystickVisitor<Joystick>> {
  template <class PyClassJoystick>
  void visit(PyClassJoystick& cl) const {
    cl.def(bp::init<>(bp::args("self"), "Default constructor."))

        .def("initialize", &Joystick::initialize, bp::args("self", "params"),
             "Initialize Joystick from Python.\n")

        .def("update_v_ref", &Joystick::update_v_ref,
             bp::args("self", "k", "gait_is_static"), "Update joystick values.")

        .def("get_p_ref", &Joystick::getPRef, bp::args("self"),
             "Get Reference Position")
        .def("get_v_ref", &Joystick::getVRef, bp::args("self"),
             "Get Reference Velocity")
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

  static void expose() {
    bp::class_<Joystick>("Joystick", bp::no_init)
        .def(JoystickVisitor<Joystick>());
  }
};

void exposeJoystick() { JoystickVisitor<Joystick>::expose(); }
