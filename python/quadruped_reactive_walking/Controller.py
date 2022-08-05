import time

import numpy as np
import pinocchio as pin
import pybullet as pyb

from . import WB_MPC_Wrapper
from .WB_MPC.Target import Target

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


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


class Interpolator:
    def __init__(self, params, x0):
        self.dt = params.dt_mpc
        self.type = params.interpolation_type

        if self.type == 3:
            self.ts = np.repeat(np.linspace(0, 2 * self.dt, 3), 2)

        self.update(x0, x0)

    def update(self, x0, x1, x2=None):
        self.q0 = x0[:3]
        self.q1 = x1[:3]
        self.v0 = x0[3:]
        self.v1 = x1[3:]
        if self.type == 0:  # Linear
            self.alpha = 0.0
            self.beta = self.v1
            self.gamma = self.q0
        elif self.type == 1:  # Quadratic fixed velocity
            self.alpha = 2 * (self.q1 - self.q0 - self.v0 * self.dt) / self.dt**2
            self.beta = self.v0
            self.gamma = self.q0
        elif self.type == 2:  # Quadratic time variable
            for i in range(3):
                q0 = self.q0[i]
                v0 = self.v0[i]
                q1 = self.q1[i]
                v1 = self.v1[i]
                if (q1 == q0) or (v1 == -v0):
                    self.alpha[i] = 0.0
                    self.beta[i] = 0.0
                    self.gamma[i] = q1
                    self.delta = 1.0
                else:
                    self.alpha[i] = (v1**2 - v0**2) / (2 * (q1 - q0))
                    self.beta[i] = v0
                    self.gamma[i] = q0
                    self.delta = 2 * (q1 - q0) / (v1 + v0) / self.dt
        elif self.type == 3:  # Spline interpolation
            from scipy.interpolate import KroghInterpolator

            if x2 is not None:
                self.q2 = x2[:3]
                self.v2 = x2[3:]
                self.y = [self.q0, self.v0, self.q1, self.v1, self.q2, self.v2]
                self.krog = KroghInterpolator(self.ts, np.array(self.y))
            else:
                self.y = [self.q0, self.v0, self.q1, self.v1]
                self.krog = KroghInterpolator(self.ts[:4], np.array(self.y))

    def interpolate(self, t):
        if self.type == 3:
            q = self.krog(t)
            v = self.krog.derivative(t)
            return q, v

        if self.type == 2:
            t *= self.delta
        q = 1 / 2 * self.alpha * t**2 + self.beta * t + self.gamma
        v = self.v1 if self.type == 1 else self.alpha * t + self.beta

        return q, v

    def plot(self, n, dt):
        import matplotlib.pyplot as plt

        ts = np.linspace(0.0, 2 * self.dt, 2 * n + 1)
        plt.style.use("seaborn")
        for i in range(3):
            plt.subplot(3, 2, (i * 2) + 1)
            plt.title("Position interpolation")
            plt.plot(ts, [self.interpolate(t)[0][i] for t in ts])
            plt.scatter(y=self.q0[i], x=0.0, color="violet", marker="+")
            plt.scatter(y=self.q1[i], x=self.dt, color="violet", marker="+")
            if self.type == 3 and self.q2 is not None:
                plt.scatter(y=self.q2[i], x=2 * self.dt, color="violet", marker="+")

            plt.subplot(3, 2, (i * 2) + 2)
            plt.title("Velocity interpolation")
            plt.plot(ts, [self.interpolate(t)[1][i] for t in ts])
            plt.scatter(y=self.v0[i], x=0.0, color="violet", marker="+")
            plt.scatter(y=self.v1[i], x=self.dt, color="violet", marker="+")
            if self.type == 3 and self.v2 is not None:
                plt.scatter(y=self.v2[i], x=2 * self.dt, color="violet", marker="+")

        plt.show()


class DummyDevice:
    def __init__(self):
        self.imu = self.IMU()
        self.joints = self.Joints()
        self.base_position = np.zeros(3)
        self.base_position[2] = 0.1944
        self.b_base_velocity = np.zeros(3)
        self.baseState = tuple()
        self.baseVel = tuple()

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
    def __init__(self, pd, params, q_init, t):
        """
        Function that computes the reference control (tau, q_des, v_des and gains)

        Args:
            params (Params object): store parameters
            q_init (array): initial position of actuators
            t (float): time of the simulation
        """
        self.q_security = np.array([1.2, 2.1, 3.14] * 4)

        self.pd = pd
        self.params = params
        self.gait = np.repeat(np.array([0, 0, 0, 0]).reshape((1, 4)), self.pd.T, axis=0)
        self.q_init = pd.q0

        self.k = 0
        self.error = False
        self.initialized = False

        self.result = Result(params)
        self.result.q_des = self.pd.q0[7:].copy()
        self.result.v_des = self.pd.v0[6:].copy()

        self.target = Target(pd)
        footsteps = [self.target.footstep(t) for t in range(pd.T)]
        self.mpc = WB_MPC_Wrapper.MPC_Wrapper(pd, params, footsteps, self.gait)
        self.mpc_solved = False
        self.k_result = 0
        self.k_solve = 0
        if self.params.interpolate_mpc:
            self.interpolator = Interpolator(
                params, np.concatenate([self.result.q_des[3:6], self.result.v_des[3:6]])
            )
        try:
            file = np.load("/tmp/init_guess.npy", allow_pickle=True).item()
            self.xs_init = list(file["xs"])
            self.us_init = list(file["us"])
            print("Initial guess loaded \n")
        except:
            self.xs_init = None
            self.us_init = None
            print("No initial guess\n")

        device = DummyDevice()
        device.joints.positions = q_init
        self.axs = None
        self.compute(device)

    def compute(self, device, qc=None):
        """
        Run one iteration of the main control loop

        Args:
            device (object): Interface with the masterboard or the simulation
        """
        t_start = time.time()

        m = self.read_state(device)
        t_measures = time.time()
        self.t_measures = t_measures - t_start

        if self.k % self.pd.mpc_wbc_ratio == 0:
            self.target.shift()
            if self.mpc_solved:
                self.k_solve = self.k
                self.mpc_solved = False

            try:
                footstep = self.target.footstep(self.pd.T)
                # self.mpc.solve(self.k, m["x_m"], self.xs_init, self.us_init)
                if self.initialized:
                    self.mpc.solve(
                        self.k,
                        self.mpc_result.xs[1],
                        footstep,
                        self.gait,
                        self.xs_init,
                        self.us_init,
                    )
                else:
                    self.mpc.solve(
                        self.k,
                        m["x_m"],
                        footstep,
                        self.gait,
                        self.xs_init,
                        self.us_init,
                    )
            except ValueError:
                self.error = True
                print("MPC Problem")

        t_mpc = time.time()
        self.t_mpc = t_mpc - t_measures

        if not self.error:
            self.mpc_result = self.mpc.get_latest_result()
            xs = self.mpc_result.xs
            if self.mpc_result.new_result:
                self.mpc_solved = True
                self.k_new = self.k
                # print(f"MPC solved in {self.k - self.k_solve} iterations")
                # self.axs = self.plot_mpc()

            if not self.initialized and self.params.save_guess:
                self.save_guess()

            self.result.FF = self.params.Kff_main * np.ones(12)
            self.result.tau_ff[3:6] = self.compute_torque(m)[:]

            # if self.params.interpolate_mpc:
            #     if self.mpc_result.new_result:
            #         if self.params.interpolation_type == 3:
            #             self.interpolator.update(xs[0], xs[1], xs[2])
            #         # self.interpolator.plot(self.pd.mpc_wbc_ratio, self.pd.dt_wbc)

            #     t = (self.k - self.k_solve + 1) * self.pd.dt_wbc
            #     q, v = self.interpolator.interpolate(t)
            # else:
            #     q, v = self.integrate_x(m)
            q = xs[1][:3]
            v = xs[1][3:]

            self.result.q_des[3:6] = q[:]
            self.result.v_des[3:6] = v[:]

            if self.axs is not None:
                [
                    self.axs[2].scatter(
                        y=self.result.tau_ff[3 + i],
                        x=(self.k - self.k_solve + 1) * self.pd.dt_wbc,
                        marker="+",
                        color=COLORS[i],
                    )
                    for i in range(3)
                ]
            self.xs_init = self.mpc_result.xs[1:] + [self.mpc_result.xs[-1]]
            self.us_init = self.mpc_result.us[1:] + [self.mpc_result.us[-1]]

        t_send = time.time()
        self.t_send = t_send - t_mpc

        # self.clamp_result(device)
        # self.security_check(m)

        if self.error:
            self.set_null_control()

        # self.pyb_camera(device)

        self.t_loop = time.time() - t_start
        self.k += 1
        self.initialized = True

        return self.error

    def pyb_camera(self, device):
        """
        Update position of PyBullet camera on the robot position to do as if it was
        attached to the robot
        """
        if self.k > 10 and self.params.enable_pyb_GUI:
            pyb.resetDebugVisualizerCamera(
                cameraDistance=0.6,
                cameraYaw=45,
                cameraPitch=-39.9,
                cameraTargetPosition=[device.height[0], device.height[1], 0.0],
            )

    def security_check(self, m):
        """
        Check if the command is fine and set the command to zero in case of error
        """

        if not self.error:
            if (np.abs(m["qj_m"]) > self.q_security).any():
                print("-- POSITION LIMIT ERROR --")
                print(m["qj_m"])
                print(np.abs(m["qj_m"]) > self.q_security)
                self.error = True
            elif (np.abs(m["vj_m"]) > 1000 * np.pi / 180).any():
                print("-- VELOCITY TOO HIGH ERROR --")
                print(m["vj_m"])
                print(np.abs(m["vj_m"]) > 1000 * np.pi / 180)
                self.error = True
            elif (np.abs(self.result.FF) > 3.2).any():
                print("-- FEEDFORWARD TORQUES TOO HIGH ERROR --")
                print(self.result.FF)
                print(np.abs(self.result.FF) > 3.2)
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
            if self.q_init[7 + 3 * i + 2] >= 0.0 and self.clamp(
                self.result.q_des[3 * i + 2], knee_min
            ):
                print("Clamping knee n " + str(i))
                self.error = set_error
            elif self.q_init[7 + 3 * i + 2] <= 0.0 and self.clamp(
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

            if self.clamp(self.result.tau_ff[i], -3.2, 3.2):
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

    def save_guess(self):
        """
        Save the result of the MPC in a file called /tmp/init_guess.npy
        """
        np.save(
            open("/tmp/init_guess.npy", "wb"),
            {"xs": self.mpc_result.xs, "us": self.mpc_result.us},
        )
        print("Initial guess saved")

    def read_state(self, device):
        qj_m = device.joints.positions
        vj_m = device.joints.velocities
        x_m = np.concatenate([qj_m[3:6], vj_m[3:6]])
        return {"qj_m": qj_m, "vj_m": vj_m, "x_m": x_m}

    def compute_torque(self, m):
        """
        Compute the feedforward torque using ricatti gains
        """
        # x_diff = np.concatenate(
        #     [
        #         pin.difference(
        #             self.pd.model,
        #             m["x_m"][: self.pd.nq],
        #             self.mpc_result.xs[0][: self.pd.nq],
        #         ),
        #         m["x_m"][self.pd.nq :] - self.mpc_result.xs[0][self.pd.nq :],
        #     ]
        # )
        # x_diff = self.mpc_result.xs[0] - m["x_m"]
        # tau = self.mpc_result.us[0] + np.dot(self.mpc_result.K[0], x_diff)
        tau = self.mpc_result.us[0]
        return tau

    def integrate_x(self, m):
        """
        Integrate the position and velocity using the acceleration computed from the
        feedforward torque
        """
        q0 = m["qj_m"][3:6].copy()
        v0 = m["vj_m"][3:6].copy()
        tau = self.result.tau_ff[3:6].copy()

        a = pin.aba(self.pd.model, self.pd.rdata, q0, v0, tau)

        v = v0 + a * self.params.dt_wbc
        q = q0 + v * self.params.dt_wbc

        return q, v

    def plot_mpc(self):
        import matplotlib.pyplot as plt

        plt.show()

        legend = ["Hip", "Shoulder", "Knee"]
        fig, axs = plt.subplots(3)
        [axs[0].plot(np.array(self.mpc_result.xs)[:, joint]) for joint in range(3)]
        axs[0].legend(legend)

        [axs[1].plot(np.array(self.mpc_result.xs)[:, 3 + joint]) for joint in range(3)]
        axs[1].legend(legend)

        [axs[2].plot(np.array(self.mpc_result.us)[:, joint]) for joint in range(3)]
        axs[2].legend(legend)

        return axs
