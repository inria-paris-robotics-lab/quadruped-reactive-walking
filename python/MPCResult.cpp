#include "qrw/MPCResult.hpp"

#include "bindings/python.hpp"

namespace detail {

/// Util class to raise error when stl container member is resized.
template <typename Class, typename vector_type>
struct non_resizable_vec_member {
 private:
  vector_type Class::*m_which;

 public:
  non_resizable_vec_member(vector_type Class::*which) : m_which(which) {}

  void operator()(Class &c, const vector_type &rhs) {
    auto &lhs = c.*m_which;
    if (lhs.size() != rhs.size()) {
      std::ostringstream ss;
      ss << "vector size is non-mutable (";
      ss << "expected " << lhs.size();
      ss << ", got " << rhs.size() << ").";
      PyErr_SetString(PyExc_ValueError, ss.str().c_str());
      bp::throw_error_already_set();
    }
    lhs = rhs;
  }
};
}  // namespace detail

template <typename Class, typename vector_type>
bp::object make_non_resizable_vec_member(vector_type Class::*which) {
  return bp::make_function(
      detail::non_resizable_vec_member<Class, vector_type>(which),
      bp::default_call_policies(),
      boost::mpl::vector3<void, Class &, const vector_type &>());
}

void exposeMPCResult() {
  bp::class_<MPCResult>(
      "MPCResult", "MPC result struct.",
      bp::init<uint, uint, uint, uint, bp::optional<uint>>(
          bp::args("self", "Ngait", "nx", "nu", "ndx", "window_size")))
      .def_readwrite("gait", &MPCResult::gait)
      .add_property("xs", bp::make_getter(&MPCResult::xs),
                    make_non_resizable_vec_member(&MPCResult::xs),
                    "Predicted future trajectory.")
      .add_property("us", bp::make_getter(&MPCResult::us),
                    make_non_resizable_vec_member(&MPCResult::us),
                    "Predicted feedforward controls.")
      .add_property("K", bp::make_getter(&MPCResult::Ks),
                    make_non_resizable_vec_member(&MPCResult::Ks),
                    "Feedback gains for the controller.")
      .def_readwrite("solving_duration", &MPCResult::solving_duration)
      .def_readonly("window_size", &MPCResult::get_window_size,
                    "Size of the window to pass back to the MPC controller.")
      .def_readwrite("num_iters", &MPCResult::num_iters)
      .def_readwrite("new_result", &MPCResult::new_result);
}
