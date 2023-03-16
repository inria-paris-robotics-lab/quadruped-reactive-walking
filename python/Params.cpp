#include "qrw/Params.hpp"

#include "bindings/python.hpp"

namespace qrw {

struct params_pickle_suite : bp::pickle_suite {
  static bp::tuple getinitargs(Params const &) { return bp::tuple(); }

  static bp::tuple getstate(bp::object obj) {
    const Params &p = bp::extract<Params const &>(obj)();
    return bp::make_tuple(p.raw_str);
  }

  static void setstate(bp::object obj, bp::tuple state) {
    Params &p = bp::extract<Params &>(obj)();
    auto str = bp::extract<std::string>(state[0])();
    p.initialize_from_str(str);
  }
};

}  // namespace qrw

constexpr auto rvp_by_value = bp::return_value_policy<bp::return_by_value>();

void exposeParams() {
  bp::class_<Params>("Params", bp::init<>("self"))
      .def("create_from_file", &Params::create_from_file,
           (bp::arg("file_path") = WALK_PARAMETERS_YAML),
           "Create Params from Python with yaml file.\n")
      .staticmethod("create_from_file")
      .def("create_from_str", &Params::create_from_str, bp::arg("content"),
           "Create Params from Python with a yaml string.\n")
      .staticmethod("create_from_str")
      .def("initialize_from_file", &Params::initialize_from_file,
           bp::args("self", "file_path"),
           "Initialize Params from Python with yaml file.\n")
      .def("initialize_from_str", &Params::initialize_from_file,
           bp::args("self", "content"),
           "Initialize Params from Python with a yaml string.\n")

      .def_pickle(qrw::params_pickle_suite())
      // Read Params from Python
      .def_readonly("raw_str", &Params::raw_str)
      .def_readwrite("config_file", &Params::config_file)
      .def_readwrite("interface", &Params::interface)
      .def_readwrite("DEMONSTRATION", &Params::DEMONSTRATION)
      .def_readwrite("SIMULATION", &Params::SIMULATION)
      .def_readwrite("LOGGING", &Params::LOGGING)
      .def_readwrite("PLOTTING", &Params::PLOTTING)
      .def_readwrite("dt_wbc", &Params::dt_wbc)
      .def_readwrite("env_id", &Params::env_id)
      .def_readwrite("q_init", &Params::q_init)
      .def_readwrite("dt_mpc", &Params::dt_mpc)
      .def_readonly("mpc_wbc_ratio", &Params::mpc_wbc_ratio)
      .def_readwrite("N_periods", &Params::N_periods)
      .def_readwrite("N_SIMULATION", &Params::N_SIMULATION)
      .def_readwrite("use_flat_plane", &Params::use_flat_plane)
      .def_readwrite("predefined_vel", &Params::predefined_vel)
      .def_readwrite("save_guess", &Params::save_guess)
      .def_readwrite("movement", &Params::movement)
      .def_readwrite("interpolate_mpc", &Params::interpolate_mpc)
      .def_readwrite("interpolation_type", &Params::interpolation_type)
      .def_readwrite("closed_loop", &Params::closed_loop)
      .def_readwrite("kf_enabled", &Params::kf_enabled)
      .def_readwrite("Kp_main", &Params::Kp_main)
      .def_readwrite("Kd_main", &Params::Kd_main)
      .def_readwrite("Kff_main", &Params::Kff_main)
      .def_readwrite("starting_nodes", &Params::starting_nodes)
      .def_readwrite("ending_nodes", &Params::ending_nodes)
      .def_readwrite("gait_repetitions", &Params::gait_repetitions)
      .def_readonly("gait", &Params::get_gait)
      .def_readonly("t_switch", &Params::get_t_switch)
      .def_readonly("v_switch", &Params::get_v_switch)
      .def("set_v_switch", &Params::set_v_switch, bp::args("self", "v_switch"),
           "Set v_switch matrix from Python.\n")
      .def_readwrite("enable_pyb_GUI", &Params::enable_pyb_GUI)
      .def_readwrite("enable_corba_viewer", &Params::enable_corba_viewer)
      .def_readwrite("enable_multiprocessing", &Params::enable_multiprocessing)
      .def_readwrite("perfect_estimator", &Params::perfect_estimator)
      .def_readwrite("use_qualisys", &Params::use_qualisys)
      .def_readwrite("ocp", &Params::ocp)
      .def_readwrite("T_gait", &Params::T_gait)
      .def_readwrite("N_gait", &Params::N_gait)
      .def_readwrite("h_ref", &Params::h_ref)
      .def_readwrite("footsteps_under_shoulders",
                     &Params::footsteps_under_shoulders)
      .add_property("task", bp::make_getter(&Params::task, rvp_by_value));

  bp::class_<OCPParams>("OCPParams", bp::no_init)
      .def_readwrite("num_threads", &OCPParams::num_threads)
      .def_readwrite("max_iter", &OCPParams::max_iter)
      .def_readwrite("init_max_iters", &OCPParams::init_max_iters)
      .def_readwrite("verbose", &OCPParams::verbose)
      .def(bp::self_ns::str(bp::self));
}
