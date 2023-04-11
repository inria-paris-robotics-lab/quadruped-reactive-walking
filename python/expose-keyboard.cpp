#include "bindings/visitors.hpp"
#include "qrw/Keyboard.hpp"

void exposeKeyboard() {
  bp::class_<qrw::KeyboardInput>("KeyboardInput", bp::no_init)
      .def(AnimatorVisitor<qrw::KeyboardInput>())
      .def("listen", &qrw::KeyboardInput::listen, bp::args("self"));
}
