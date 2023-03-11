import time

import numpy as np
import pinocchio as pin

import quadruped_reactive_walking as qrw
from . import wb_mpc
from .wb_mpc.target import Target
from .tools.Utils import init_robot, quaternionToRPY
from .wb_mpc.problem_data import TaskSpec
from .tools.Interpolator import Interpolator
from typing import Type


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
    def __init__(self, h):
        self.imu = self.IMU()
        self.joints = self.Joints()
        self.base_position = np.zeros(3)
        self.base_position[2] = 0.1944
        self.b_base_velocity = np.zeros(3)
        self.baseState = ((0.0, 0.0, h), (0.0, 0.0, 0.0, 1.0))
        self.baseVel = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

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


def make_footsteps_and_refs(params, target):
    footsteps = []
    base_refs = []
    for k in range(params.T * params.mpc_wbc_ratio):
        if params.movement == "base_circle" or params.movement == "walk":
            target_base = np.zeros(6)
            target_footstep = np.zeros((3, 4))
        else:
            target_footstep = target.compute(k).copy()
            target_base = np.array([0.0, 0.0, params.h_ref])

        if k % params.mpc_wbc_ratio == 0:
            base_refs.append(target_base.copy())
            footsteps.append(target_footstep.copy())

    return footsteps, base_refs


class Controller:
    def __init__(
        self,
        params: qrw.Params,
        q_init,
        t,
        solver_cls: Type[wb_mpc.OCPAbstract],
        solver_kwargs={},
    ):
        """
        Function that computes the reference control (tau, q_des, v_des and gains)

        Args:
            params (Params object): store parameters
            q_init (array): initial position of actuators
            t (float): time of the simulation
        """
        self.q_security = np.array([1.2, 2.1, 3.14] * 4)

        self.params = params
        init_robot(q_init, params)
        self.pd = TaskSpec(params)
        self.rdata = self.pd.create_rdata()

        self.k = 0
        self.error = False
        self.initialized = False

        if params.predefined_vel:
            self.joystick = qrw.AnimatorBase(params)
        else:
            self.joystick = qrw.Joystick(params)

        self.estimator = qrw.Estimator()
        self.estimator.initialize(params)
        self.q = np.zeros(18)

        self.result = Result(params)
        self.result.q_des = self.pd.q0[7:].copy()
        self.result.v_des = self.pd.v0[6:].copy()

        self.target = Target(params)
        self.footsteps, self.base_refs = make_footsteps_and_refs(
            self.params, self.target
        )

        if params.enable_multiprocessing:
            from .wbmpc_wrapper_multiprocess import MultiprocessMPCWrapper as MPCWrapper
        else:
            from .wbmpc_wrapper_sync import SyncMPCWrapper as MPCWrapper

        self.mpc = MPCWrapper(
            params,
            self.footsteps,
            self.base_refs,
            solver_cls=solver_cls,
            **solver_kwargs
        )
        self.gait = np.array([[1, 1, 1, 1]] * (params.T + 1))
        self.mpc_solved = False
        self.k_result = 0
        self.k_solve = 0
        if self.params.interpolate_mpc:
            self.interpolator = Interpolator(params, self.pd.x0)
        try:
            filename = np.load("/tmp/init_guess.npy", allow_pickle=True).item()
            self.xs_init = list(filename["xs"])
            self.us_init = list(filename["us"])
            print("Initial guess loaded.\n")
        except Exception:
            self.xs_init = None
            self.us_init = None
            print("No initial guess found.\n")

        self.filter_q = qrw.Filter()
        self.filter_q.initialize(params)
        self.filter_v = qrw.Filter()
        self.filter_v.initialize(params)

        device = DummyDevice(params.h_ref)
        device.joints.positions = q_init
        self.compute(device)

    def compute(self, device, qc=None):
        """
        Run one iteration of the main control loop

        Args:
            device (object): Interface with the masterboard or the simulation
        """
        t_start = time.time()

        self.joystick.update_v_ref(self.k, False)

        oRh, hRb, oTh = self.run_estimator(device)

        t_measures = time.time()
        self.t_measures = t_measures - t_start

        if self.params.movement == "base_circle" or self.params.movement == "walk":
            self.target_base = self.v_ref
            self.target_footstep = np.zeros((3, 4))
        else:
            self.target_base = np.zeros(3)
            self.target_footstep = self.target.compute(
                self.k + self.params.T * self.params.mpc_wbc_ratio
            )

        if self.k % self.params.mpc_wbc_ratio == 0:
            if self.mpc_solved:
                self.k_solve = self.k
                self.mpc_solved = False

            if self.params.closed_loop or not self.initialized:
                x = self.x_estim
            else:
                x = self.mpc_result.xs[1]

            try:
                self.mpc.solve(
                    self.k,
                    x,
                    self.target_footstep.copy(),
                    self.target_base.copy(),
                    self.xs_init,
                    self.us_init,
                )
            except ValueError:
                self.error = True
                print("MPC Problem")

        t_mpc = time.time()
        self.t_mpc = t_mpc - t_measures

        if not self.error:
            self.mpc_result: Result = self.mpc.get_latest_result()
            self.gait[:, :] = self.mpc_result.gait
            xs = self.mpc_result.xs
            if self.mpc_result.new_result:
                self.mpc_solved = True
                self.k_new = self.k

            if not self.initialized and self.params.save_guess:
                self.save_guess()

            self.result.FF = self.params.Kff_main * np.ones(12)
            self.result.tau_ff = self.compute_torque()[:]

            if self.params.interpolate_mpc:
                if self.mpc_result.new_result:
                    if self.params.interpolation_type == 3:
                        self.interpolator.update(xs[0], xs[1], xs[2])
                    # self.interpolator.plot(self.params.mpc_wbc_ratio, self.params.dt_wbc)
                t = (self.k - self.k_solve + 1) * self.params.dt_wbc
                q, v = self.interpolator.interpolate(t)
            else:
                q, v = self.integrate_x()

            self.result.q_des = q[:]
            self.result.v_des = v[:]

            self.xs_init = self.mpc_result.xs[1:] + [self.mpc_result.xs[-1]]
            self.us_init = self.mpc_result.us[1:] + [self.mpc_result.us[-1]]

        t_send = time.time()
        self.t_send = t_send - t_mpc

        self.clamp_result(device)
        self.security_check()

        if self.error:
            self.set_null_control()

        self.t_loop = time.time() - t_start
        self.k += 1
        self.initialized = True

        return self.error

    def security_check(self):
        """
        Check if the command is fine and set the command to zero in case of error
        """

        if not self.error:
            if (np.abs(self.q_estimate[7:]) > self.q_security).any():
                print("-- POSITION LIMIT ERROR --")
                print(self.q_estimate[7:])
                print(np.abs(self.q_estimate[7:]) > self.q_security)
                self.error = True
            elif (np.abs(self.v_estimate[6:]) > 100.0).any():
                print("-- VELOCITY TOO HIGH ERROR --")
                print(self.v_estimate[6:])
                print(np.abs(self.v_estimate[6:]) > 100.0)
                self.error = True
            elif (np.abs(self.result.FF) > 5.0).any():
                print("-- FEEDFORWARD TORQUES TOO HIGH ERROR --")
                print(self.result.FF)
                print(np.abs(self.result.FF) > 5.0)
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
            if self.pd.q0[7 + 3 * i + 2] >= 0.0 and self.clamp(
                self.result.q_des[3 * i + 2], knee_min
            ):
                print("Clamping knee n " + str(i))
                self.error = set_error
            elif self.pd.q0[7 + 3 * i + 2] <= 0.0 and self.clamp(
                self.result.q_des[3 * i + 2], max_value=-knee_min
            ):
                print("Clamping knee n " + str(i))
                self.error = set_error

        clamped_tau = []
        clamped_pos = []
        clamped_vel = []
        for i in range(12):
            if self.clamp(
                self.result.q_des[i],
                device.joints.positions[i] - 4.0,
                device.joints.positions[i] + 4.0,
            ):
                clamped_pos.append(i)
                self.error = set_error

            if self.clamp(
                self.result.v_des[i],
                device.joints.velocities[i] - 100.0,
                device.joints.velocities[i] + 100.0,
            ):
                clamped_vel.append(i)
                self.error = set_error

            if self.clamp(self.result.tau_ff[i], -3.2, 3.2):
                clamped_tau.append(i)
                self.error = set_error
        if len(clamped_tau) > 0:
            print("Clamping torque of motors {}".format(clamped_tau))
        if len(clamped_pos) > 0:
            print("Clamping position of motors {}".format(clamped_pos))
        if len(clamped_vel) > 0:
            print("Clamping velocity of motors {}".format(clamped_vel))

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

    def save_guess(self, filename="/tmp/init_guess.npy"):
        """
        Save the result of the MPC in a file called /tmp/init_guess.npy
        """
        np.save(
            open(filename, "wb"),
            {"xs": self.mpc_result.xs, "us": self.mpc_result.us},
        )
        print("Initial guess saved")

    def run_estimator(
        self, device, q_perfect=np.zeros(6), b_baseVel_perfect=np.zeros(3)
    ):
        """
        Call the estimator and retrieve the reference and estimated quantities.
        Run a filter on q, h_v and v_ref.

        @param device device structure holding simulation data
        @param q_perfect 6D perfect position of the base in world frame
        @param v_baseVel_perfect 3D perfect linear velocity of the base in base frame
        """
        footstep = np.array(self.params.footsteps_init.tolist())

        if self.k < 2:
            self.estimator.initialize_IMU_Yaw()

        self.estimator.run(
            self.gait,
            footstep.reshape((3, 4), order="F"),
            device.imu.linear_acceleration,
            device.imu.gyroscope,
            device.imu.attitude_euler,
            device.joints.positions,
            device.joints.velocities,
            q_perfect,
            b_baseVel_perfect,
        )

        self.estimator.update_reference_state(self.joystick.get_v_ref())

        oRh = self.estimator.get_oRh()
        hRb = self.estimator.get_hRb()
        oTh = self.estimator.get_oTh().reshape((3, 1))

        self.v_ref = self.estimator.get_base_vel_ref()
        self.h_v = self.estimator.get_h_v()
        self.h_v_windowed = self.estimator.get_h_v_filtered()
        self.v_windowed = self.estimator.get_v_filtered()

        self.q_estimate = self.estimator.get_q_estimate()
        self.v_estimate = self.estimator.get_v_estimate()

        # bp_m = np.array([e for tup in device.baseState for e in tup])
        # bv_m = np.array([e for tup in device.baseVel for e in tup])
        self.q[:3] = self.q_estimate[:3]
        self.q[3:6] = quaternionToRPY(self.q_estimate[3:7]).ravel()
        self.q[6:] = self.q_estimate[7:]

        self.v = self.estimator.get_v_reference()

        self.base_position_filtered = self.filter_q.filter(self.q[:6], True)

        self.q_filtered = self.q_estimate.copy()
        self.q_filtered[:3] = self.base_position_filtered[:3]
        self.q_filtered[3:7] = pin.Quaternion(
            pin.rpy.rpyToMatrix(self.base_position_filtered[3:])
        ).coeffs()
        self.v_filtered = self.v_estimate.copy()
        # self.v_filtered[:6] = np.zeros(6)
        # self.v_filtered[:6] = self.filter_v.filter(self.v_windowed, False)
        self.v_filtered[:6] = self.filter_v.filter(self.v_estimate[:6], False)

        self.x_estim = np.concatenate([self.q_filtered, self.v_filtered])
        return oRh, hRb, oTh

    def compute_torque(self):
        """
        Compute the feedforward torque using ricatti gains
        """
        x_diff = np.concatenate(
            [
                pin.difference(
                    self.pd.model,
                    self.x_estim[: self.pd.nq],
                    self.mpc_result.xs[0][: self.pd.nq],
                ),
                self.mpc_result.xs[0][self.pd.nq :] - self.x_estim[self.pd.nq :],
            ]
        )
        tau = self.mpc_result.us[0] + np.dot(self.mpc_result.K[0], x_diff)
        return tau

    def integrate_x(self):
        """
        Integrate the position and velocity using the acceleration computed from the
        feedforward torque
        """
        q0 = self.q_estimate.copy()
        v0 = self.v_estimate.copy()
        tau = np.concatenate([np.zeros(6), self.result.tau_ff.copy()])

        a = pin.aba(self.pd.model, self.rdata, q0, v0, tau)

        v = v0 + a * self.params.dt_wbc
        q = pin.integrate(self.pd.model, q0, v * self.params.dt_wbc)

        return q[7:], v[6:]

    def plot_mpc(self, base=False, joints=True):
        import matplotlib.pyplot as plt

        if base:
            legend = ["X", "Y", "Z"]
            _, axs = plt.subplots(2)
            [axs[0].plot(np.array(self.mpc_result.xs)[:, axis]) for axis in range(3)]
            axs[0].legend(legend)
            axs[0].set_title("Base position")

            [
                axs[1].plot(np.array(self.mpc_result.xs)[:, 19 + axis])
                for axis in range(3)
            ]
            axs[1].legend(legend)
            axs[1].set_title("Base velocity")

        if joints:
            legend = ["Hip", "Shoulder", "Knee"]
            _, axs = plt.subplots(3, 4, sharex=True)
            for foot in range(4):
                [
                    axs[0, foot].plot(
                        np.array(self.mpc_result.xs)[:, 7 + 3 * foot + joint]
                    )
                    for joint in range(3)
                ]
                axs[0, foot].legend(legend)
                axs[0, foot].set_title(
                    "Joint positions for " + self.pd.feet_names[foot]
                )

                [
                    axs[1, foot].plot(
                        np.array(self.mpc_result.xs)[:, 19 + 6 + 3 * foot + joint]
                    )
                    for joint in range(3)
                ]
                axs[1, foot].legend(legend)
                axs[1, foot].set_title("Joint velocity for " + self.pd.feet_names[foot])

                [
                    axs[2, foot].plot(np.array(self.mpc_result.us)[:, 3 * foot + joint])
                    for joint in range(3)
                ]
                axs[2, foot].legend(legend)
                axs[2, foot].set_title(
                    "Joint torques for foot " + self.pd.feet_names[foot]
                )

        plt.show()