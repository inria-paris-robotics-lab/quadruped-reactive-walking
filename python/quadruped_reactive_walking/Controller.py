import time

import numpy as np
import pinocchio as pin
import pybullet as pyb

from . import WB_MPC_Wrapper
from .solo3D.utils import quaternionToRPY
from .tools.utils_mpc import init_robot


class Result:
    """
    Object to store the result of the control loop
    It contains what is sent to the robot (gains, desired positions and velocities,
    feedforward torques)
    """

    def __init__(self, params):
        self.P = np.array(params.Kp_main.tolist() * 4)
        self.D = np.array(params.Kd_main.tolist() * 4)
        self.FF = params.Kff_main * np.ones(12)
        self.q_des = np.zeros(12)
        self.v_des = np.zeros(12)
        self.tau_ff = np.zeros(12)


class DummyDevice:
    def __init__(self):
        self.imu = self.IMU()
        self.joints = self.Joints()
        self.base_position = np.zeros(3)
        self.base_position[2] = 0.1944
        self.b_base_velocity = np.zeros(3)

    class IMU:
        def __init__(self):
            self.linear_acceleration = np.zeros(3)
            self.gyroscope = np.zeros(3)
            self.attitude_euler = np.zeros(3)
            self.attitude_quaternion = np.zeros(4)

    class Joints:
        def __init__(self):
            self.positions = np.zeros(12)
            self.velocities = np.zeros(12)


class Controller:
    def __init__(self, params, q_init, t):
        """Function that runs a simulation scenario based on a reference velocity profile, an environment and
        various parameters to define the gait

        Args:
            params (Params object): store parameters
            q_init (array): initial position of actuators
            t (float): time of the simulation
        """
        self.enable_pyb_GUI = params.enable_pyb_GUI
        self.q_security = np.array([1.2, 2.1, 3.14] * 4)

        self.mpc = WB_MPC_Wrapper.MPC_Wrapper(params)

        self.error = False
        self.result = Result(params)

        device = DummyDevice()
        device.joints.positions = q_init
        self.compute(device)

    def compute(self, device, qc=None):
        """Run one iteration of the main control loop

        Args:
            device (object): Interface with the masterboard or the simulation
        """
        t_start = time.time()

        try:
            self.mpc_wrapper.solve()
        except ValueError:
            print("MPC Problem")

        self.mpc_result, self.mpc_cost = self.mpc_wrapper.get_latest_result()

        if not self.error:
            self.result.P = np.array(self.params.Kp_main.tolist() * 4)
            self.result.D = np.array(self.params.Kd_main.tolist() * 4)
            self.result.FF = self.params.Kff_main * np.ones(12)
            self.result.q_des = np.zeros(12)
            self.result.v_des = np.zeros(12)
            self.result.tau_ff = np.zeros(12)

        self.t_wbc = time.time() - t_start

        self.clamp_result(device)
        self.security_check()
        if self.error or self.joystick.get_stop():
            self.set_null_control()
        
        self.pyb_camera(device)

        self.t_loop = time.time() - t_start
        self.k += 1

        return self.error

    def pyb_camera(self, device):
        """
        Update position of PyBullet camera on the robot position to do as if it was
        attached to the robot
        """
        if self.k > 10 and self.enable_pyb_GUI:
            pyb.resetDebugVisualizerCamera(
                cameraDistance=0.6,
                cameraYaw=45,
                cameraPitch=-39.9,
                cameraTargetPosition=[device.height[0], device.height[1], 0.0],
            )

    def security_check(self):
        """
        Check if the command is fine and set the command to zero in case of error
        """

        if not (self.error or self.joystick.get_stop()):
            if (np.abs(self.estimator.get_q_estimate()[7:]) > self.q_security).any():
                print("-- POSITION LIMIT ERROR --")
                print(self.estimator.get_q_estimate()[7:])
                print(np.abs(self.estimator.get_q_estimate()[7:]) > self.q_security)
                self.error = True
            elif (np.abs(self.estimator.get_v_security()) > 100.0).any():
                print("-- VELOCITY TOO HIGH ERROR --")
                print(self.estimator.get_v_security())
                print(np.abs(self.estimator.get_v_security()) > 100.0)
                self.error = True
            elif (np.abs(self.wbcWrapper.tau_ff) > 8.0).any():
                print("-- FEEDFORWARD TORQUES TOO HIGH ERROR --")
                print(self.wbcWrapper.tau_ff)
                print(np.abs(self.wbcWrapper.tau_ff) > 8.0)
                self.error = True

    def clamp(self, num, min_value=None, max_value=None):
        clamped = False
        if min_value is not None and num <= min_value:
            num = min_value
            clamped = True
        if max_value is not None and num >= max_value:
            num = max_value
            clamped = True
        return clamped

    def clamp_result(self, device, set_error=False):
        """
        Clamp the result
        """
        hip_max = 120.0 * np.pi / 180.0
        knee_min = 5.0 * np.pi / 180.0
        for i in range(4):
            if self.clamp(self.result.q_des[3 * i + 1], -hip_max, hip_max):
                print("Clamping hip n " + str(i))
                self.error = set_error
            if self.q_init[6 + 3 * i + 2] >= 0.0 and self.clamp(
                self.result.q_des[3 * i + 2], knee_min
            ):
                print("Clamping knee n " + str(i))
                self.error = set_error
            elif self.q_init[6 + 3 * i + 2] <= 0.0 and self.clamp(
                self.result.q_des[3 * i + 2], max_value=-knee_min
            ):
                print("Clamping knee n " + str(i))
                self.error = set_error

        for i in range(12):
            if self.clamp(
                self.result.q_des[i],
                device.joints.positions[i] - 4.0,
                device.joints.positions[i] + 4.0,
            ):
                print("Clamping position difference of motor n " + str(i))
                self.error = set_error

            if self.clamp(
                self.result.v_des[i],
                device.joints.velocities[i] - 100.0,
                device.joints.velocities[i] + 100.0,
            ):
                print("Clamping velocity of motor n " + str(i))
                self.error = set_error

            if self.clamp(self.result.tau_ff[i], -8.0, 8.0):
                print("Clamping torque of motor n " + str(i))
                self.error = set_error

    def set_null_control(self):
        """
        Send default null values to the robot
        """
        self.result.P = np.zeros(12)
        self.result.D = 0.1 * np.ones(12)
        self.result.FF = np.zeros(12)
        self.result.q_des[:] = np.zeros(12)
        self.result.v_des[:] = np.zeros(12)
        self.result.tau_ff[:] = np.zeros(12)
