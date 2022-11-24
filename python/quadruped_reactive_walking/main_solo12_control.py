import threading
import time
from pathlib import Path
import numpy as np
import git
from datetime import datetime

import quadruped_reactive_walking as qrw
from .Controller import Controller
from .tools.LoggerControl import LoggerControl

from typing import Type, Literal
from .WB_MPC import CrocOCP, ProxOCP
import tap
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="croc", choices=["croc", "prox"])
    return parser.parse_args()


params = qrw.Params()  # Object that holds all controller parameters

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
msg = repo.head.object.message + "\nCommit: " + sha

if params.SIMULATION:
    from .tools.PyBulletSimulator import PyBulletSimulator
else:
    import libodri_control_interface_pywrap as oci
    from .tools.qualisysClient import QualisysClient


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

    i = threading.Thread(target=get_input)
    i.start()
    print("Put the robot on the floor and press Enter")

    while i.is_alive():
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


def damp_control(device, nb_motors):
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


def control_loop(args):
    """
    Main function that calibrates the robot, get it into a default waiting position then launch
    the main control loop once the user has pressed the Enter key

    Args:
        des_vel_analysis (string)
    """
    if not params.SIMULATION:
        params.enable_pyb_GUI = False

    # Default position after calibration
    q_init = np.array(params.q_init.tolist())
    if args.solver == "croc":
        solver_t = CrocOCP
    else:
        solver_t = ProxOCP
    controller = Controller(params, q_init, 0.0, solver_t)

    if params.SIMULATION:
        device = PyBulletSimulator()
        qc = None
    else:
        device = oci.robot_from_yaml_file(params.config_file)
        qc = QualisysClient(ip="140.93.16.160", body_id=0)

    if params.LOGGING or params.PLOTTING:
        loggerControl = LoggerControl(
            controller.pd, params, log_size=params.N_SIMULATION
        )

    if params.SIMULATION:
        device.Init(
            q_init,
            params.envID,
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
    while (not device.is_timeout) and (t < t_max) and (not controller.error):
        t_start_whole = time.time()

        device.parse_sensor_data()
        if controller.compute(device, qc):
            break

        if t <= 10 * params.dt_wbc and check_position_error(device, controller):
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
            loggerControl.sample(controller, device, qc)

        t_end_whole = time.time()

        t += params.dt_wbc

        dT_whole = T_whole
        T_whole = time.time()
        dT_whole = T_whole - dT_whole

        t_log_whole[k_log_whole] = t_end_whole - t_start_whole
        k_log_whole += 1

    # ****************************************************************
    damp_control(device, 12)

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
        date_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
        log_path = Path("/tmp") / "logs" / date_str
        log_path.mkdir(parents=True)
        loggerControl.save(str(log_path))
        with open(str(log_path / "readme.txt"), "w") as f:
            f.write(msg)

        if params.PLOTTING:
            loggerControl.plot(save=True, fileName=str(log_path))
            print("Plots saved in ", str(log_path) + "/")

            plt.show()

    if params.SIMULATION and params.enable_pyb_GUI:
        device.Stop()

    print("End of script")
    return loggerControl


if __name__ == "__main__":
    import os
    os.nice(-20)

    args = parse_args()
    log = control_loop(args)
