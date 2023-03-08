import threading
import time
import numpy as np
import git
import quadruped_reactive_walking as qrw

from datetime import datetime

from quadruped_reactive_walking.controller import Controller
from quadruped_reactive_walking.tools.logger_control import (
    LoggerControl,
    TEMP_DIRNAME,
    DATE_STRFORMAT,
)
from quadruped_reactive_walking.wb_mpc import CrocOCP, AlgtrOCP

import tqdm
import argparse
import matplotlib.pyplot as plt
import enum
import colorama

from colorama import Fore

colorama.init()

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["lines.linewidth"] = 1.0


class SolverChoices(enum.Enum):
    croc = "croc"
    prox = "prox"
    fddp = "fddp"

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--solver",
        choices=list(SolverChoices),
        type=SolverChoices,
        required=True,
        help="Solver choice. Default: %(default)s.",
    )
    parser.add_argument("--run_croc", action="store_true")
    return parser.parse_args()


params = qrw.Params()  # Object that holds all controller parameters

try:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    msg = repo.head.object.message + "\nCommit: " + sha
except ValueError:
    msg = "\nCommit not found"


def get_input():
    """
    Thread to get the input
    """
    input()


def put_on_the_floor(device, q_init):
    """
    Make the robot go to the default initial position and wait for the user
    to press the Enter key to start the main control loop

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        q_init (array): the default position of the robot
    """
    print("PUT ON THE FLOOR.")

    Kp_pos = 6.0
    Kd_pos = 0.3

    device.joints.set_position_gains(Kp_pos * np.ones(12))
    device.joints.set_velocity_gains(Kd_pos * np.ones(12))
    device.joints.set_desired_positions(q_init)
    device.joints.set_desired_velocities(np.zeros(12))
    device.joints.set_torques(np.zeros(12))

    thread = threading.Thread(target=get_input)
    thread.start()
    print("Put the robot on the floor and press Enter")

    while thread.is_alive():
        device.parse_sensor_data()
        device.send_command_and_wait_end_of_cycle(params.dt_wbc)

    # Slow increase till 1/4th of mass is supported by each foot
    duration_increase = 2.0
    steps = int(duration_increase / params.dt_wbc)
    tau_ff = np.array([0.0, 0.022, 0.5] * 2 + [0.0, -0.022, -0.5] * 2)
    # tau_ff = np.array([0.0, 0.022, 0.5, 0.0, 0.022, 0.5, 0.0, 0.025, 0.575, 0.0, 0.025, 0.575])

    for i in range(steps):
        device.joints.set_torques(tau_ff * i / steps)
        device.parse_sensor_data()
        device.send_command_and_wait_end_of_cycle(params.dt_wbc)

    print("Start the motion.")


def check_position_error(device, controller):
    """
    Check the distance between current and desired position of the joints

    Args:
        device (robot wrapper): a wrapper to communicate with the robot
        controller (array): the controller storing the desired position
    """
    if np.max(np.abs(controller.result.q_des - device.joints.positions)) > 0.15:
        print("DIFFERENCE: ", controller.result.q_des - device.joints.positions)
        print("q_des: ", controller.result.q_des)
        print("q_mes: ", device.joints.positions)
        return True
    return False


def damp_controls(device, nb_motors):
    """
    Damp the control during 2.5 seconds

    Args:
        device  (robot wrapper): a wrapper to communicate with the robot
        nb_motors (int): number of motors
    """
    t = 0.0
    t_max = 2.5
    while (not device.is_timeout) and (t < t_max):
        device.parse_sensor_data()

        # Set desired quantities for the actuators
        device.joints.set_position_gains(np.zeros(nb_motors))
        device.joints.set_velocity_gains(0.1 * np.ones(nb_motors))
        device.joints.set_desired_positions(np.zeros(nb_motors))
        device.joints.set_desired_velocities(np.zeros(nb_motors))
        device.joints.set_torques(np.zeros(nb_motors))

        # Send command to the robot
        device.send_command_and_wait_end_of_cycle(params.dt_wbc)
        if (t % 1) < 5e-5:
            print("IMU attitude:", device.imu.attitude_euler)
            print("joint pos   :", device.joints.positions)
            print("joint vel   :", device.joints.velocities)
            device.robot_interface.PrintStats()

        t += params.dt_wbc


def get_device(is_simulation: bool) -> tuple:
    if is_simulation:
        from .tools.pybullet_sim import PyBulletSimulator

        device = PyBulletSimulator()
        qc = None
    else:
        import libodri_control_interface_pywrap as oci

        device = oci.robot_from_yaml_file(params.config_file)

        if params.use_qualisys:
            from .tools.qualisys_client import QualisysClient

            qc = QualisysClient(ip="140.93.16.160", body_id=0)
        else:
            qc = None
    return (device, qc)


def main(args):
    """
    Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key

    Args:
        des_vel_analysis (string)
    """
    if not params.SIMULATION:
        params.enable_pyb_GUI = False

    solver_kwargs = {}

    # Default position after calibration
    q_init = np.array(params.q_init.tolist())
    if args.solver == SolverChoices.croc:
        solver_cls = CrocOCP
    else:
        solver_cls = AlgtrOCP
        solver_kwargs["run_croc"] = args.run_croc
        solver_kwargs["use_prox"] = args.solver == SolverChoices.prox

    controller = Controller(
        params, q_init, 0.0, solver_cls, solver_kwargs=solver_kwargs
    )
    device, qc = get_device(params.SIMULATION)

    # viewer = meshcat_viewer.MeshcatViewer(controller.pd.robot)

    if params.LOGGING or params.PLOTTING:
        logger = LoggerControl(params, log_size=params.N_SIMULATION)
    else:
        logger = None

    if params.SIMULATION:
        device.Init(
            q_init,
            params.env_id,
            params.use_flat_plane,
            params.enable_pyb_GUI,
            params.dt_wbc,
        )
    else:
        device.initialize(q_init[:])
        device.joints.set_zero_commands()
        device.parse_sensor_data()
        put_on_the_floor(device, q_init)

    # CONTROL LOOP ***************************************************
    t = 0.0
    t_max = (params.N_SIMULATION - 1) * params.dt_wbc

    t_log_whole = np.zeros((params.N_SIMULATION))
    k_log_whole = 0
    T_whole = time.time()
    dT_whole = 0.0
    disable = params.ocp.verbose
    bar_format = (
        "{desc}: {percentage:.4f}%|{bar}| {n:.3f}/{total:.3f} [{elapsed}<{remaining}]"
    )
    with tqdm.tqdm(
        desc="MPC cycles",
        total=t_max + params.dt_wbc,
        unit="s",
        disable=disable,
        bar_format=bar_format,
    ) as prog_bar:
        while (not device.is_timeout) and (t < t_max) and (not controller.error):
            t_start_whole = time.time()

            device.parse_sensor_data()
            if controller.compute(device, qc):
                break

            if t <= 10 * params.dt_wbc and check_position_error(device, controller):
                print(
                    Fore.YELLOW + "Position error encountered; breaking." + Fore.RESET
                )
                break

            device.joints.set_position_gains(controller.result.P)
            device.joints.set_velocity_gains(controller.result.D)
            device.joints.set_desired_positions(controller.result.q_des)
            device.joints.set_desired_velocities(controller.result.v_des)
            device.joints.set_torques(
                controller.result.FF * controller.result.tau_ff.ravel()
            )
            device.send_command_and_wait_end_of_cycle(params.dt_wbc)

            if params.LOGGING or params.PLOTTING:
                logger.sample(controller, device, qc)

            t_end_whole = time.time()

            t += params.dt_wbc

            dT_whole = T_whole
            T_whole = time.time()
            dT_whole = T_whole - dT_whole

            t_log_whole[k_log_whole] = t_end_whole - t_start_whole
            k_log_whole += 1
            prog_bar.update(params.dt_wbc)

    # ****************************************************************
    damp_controls(device, 12)

    if params.enable_multiprocessing:
        print("Stopping parallel process MPC")
        controller.mpc.stop_parallel_loop()

    # ****************************************************************

    # Send 0 torques to the motors.
    device.joints.set_torques(np.zeros(12))
    device.send_command_and_wait_end_of_cycle(params.dt_wbc)

    if device.is_timeout:
        print("Masterboard timeout detected.")

    if params.LOGGING:
        date_str = datetime.now().strftime(DATE_STRFORMAT)
        log_path = TEMP_DIRNAME / "logs" / date_str
        log_path.mkdir(parents=True, exist_ok=True)
        logger.save(str(log_path))
        with open(str(log_path / "readme.txt"), "w") as f:
            f.write(msg)

        if params.PLOTTING:
            logger.plot(save=True, filename=str(log_path))
            print("Plots saved in ", str(log_path) + "/")

            mpc = controller.mpc

            def plot_prox_ocp(ocp: AlgtrOCP):
                nplt = 3
                if args.run_croc:
                    nplt += 1
                h = 7.2
                fig, axs = plt.subplots(nplt, 1, figsize=(6.4, h), layout="constrained")
                fig: plt.Figure
                fig.supxlabel("MPC cycle $k$")

                plt.sca(axs[0])
                plt.plot(ocp.prox_stops, label="algtr", ls="-")
                plt.plot(ocp.croc_stops, label="croco", ls="dotted")
                plt.yscale("log")
                plt.title("$\\ell_\\infty$-norm of stopping criterion")
                plt.legend()

                plt.sca(axs[1])
                plt.plot(ocp.prox_stops_2, label="algtr", ls="-")
                plt.plot(ocp.croc_stops_2, label="croco", ls="dotted")
                plt.title("Squared norm stopping criterion")
                plt.yscale("log")
                plt.legend()

                plt.sca(axs[2])
                plt.plot(ocp.prox_iters, label="algtr", ls="-")
                plt.plot(ocp.croc_iters, label="croco", ls="dotted")
                plt.title("Number of OCP iterations")
                plt.grid(visible=True, which="minor", axis="y")
                plt.legend()

                if args.run_croc:
                    plt.sca(axs[3])
                    plt.plot(ocp.x_solver_errs, label="state $x$", c="b")
                    plt.plot(ocp.u_solver_errs, label="control $u$", c="r", ls="dotted")
                    plt.plot(ocp.fb_errs, label="feedbacks $K$", c="g", ls="dotted")
                    plt.legend()
                    plt.yscale("log")
                    plt.title("$\\ell_\\infty$ error between solver solutions")

            if hasattr(mpc, "ocp") and isinstance(mpc.ocp, AlgtrOCP):
                plot_prox_ocp(mpc.ocp)

            plt.show()

    if params.SIMULATION and params.enable_pyb_GUI:
        device.Stop()

    print("End of script")


if __name__ == "__main__":
    import cProfile
    import pstats

    args = parse_args()
    profiler = cProfile.Profile()
    profiler.enable()
    main(args)
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    stats.dump_stats("stats.prof")
    stats.print_stats()
