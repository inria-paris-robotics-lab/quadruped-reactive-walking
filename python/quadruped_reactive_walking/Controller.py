import time

import numpy as np
import pinocchio as pin
import pybullet as pyb

from . import WB_MPC_Wrapper


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
90

class Controller:
    def __init__(self, pd, target, params, q_init, t):
        """Function that runs a simulation scenario based on a reference velocity profile, an environment and
        various parameters to define the gait

        Args:
            params (Params object): store parameters
            q_init (array): initial position of actuators
            t (float): time of the simulation
        """
        self.q_security = np.array([1.2, 2.1, 3.14] * 4)

        self.mpc = WB_MPC_Wrapper.MPC_Wrapper(pd, target, params)
        self.pd = pd
        self.target = target
        self.point_target = []
        self.params = params
        self.q_init = pd.q0

        self.k = 0
        self.cnt_mpc = 0
        self.cnt_wbc = 0
        self.error = False
        self.initialized = False
        self.interpolated = False
        self.result = Result(params)
        self.q = self.pd.q0[7:].copy()
        self.v = self.pd.v0[6:].copy()

        device = DummyDevice()
        device.joints.positions = q_init
        try:
            file = np.load("/tmp/init_guess.npy", allow_pickle=True).item()
            self.xs_init = list(file["xs"])
            self.us_init = list(file["us"])
            print("Initial guess loaded \n")
        except:
            self.xs_init = None
            self.us_init = None
            print("No initial guess\n")

        # self.compute(device)

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

        self.point_target = self.target.evaluate_in_t(1)[self.pd.rfFootId]

        if self.k % self.pd.r1 == 0:
            try:
                self.target.update(self.cnt_mpc)
                self.target.shift_gait()
                self.interpolated = False
                self.cnt_wbc = 0

                # Closed-loop
                self.mpc.solve(self.k, m["x_m"], self.xs_init, self.us_init)

                # Trajectory tracking
                # if self.initialized:
                    # self.mpc.solve(
                        # self.k, self.mpc_result.xs[1], self.xs_init, self.us_init)
                # else:
                    # self.mpc.solve(self.k, m["x_m"],
                                #    self.xs_init, self.us_init)

                self.cnt_mpc += 1        
            except ValueError:
                self.error = True
                print("MPC Problem")

        t_mpc = time.time()
        self.t_mpc = t_mpc - t_measures

        if not self.error:
            self.mpc_result = self.mpc.get_latest_result()

            # ## ONLY IF YOU WANT TO STORE THE FIRST SOLUTION TO WARM-START THE INITIAL Problem ###
            #if not self.initialized:
            #   np.save(open('/tmp/init_guess.npy', "wb"), {"xs": self.mpc_result.xs, "us": self.mpc_result.us} )
            #   print("Initial guess saved")

            # Keep only the actuated joints and set the other to default values
            if not self.interpolated:
                self.q_interpolated, self.v_interpolated = self.interpolate_traj(device,\
                                            np.array(self.mpc_result.xs)[1, :self.pd.nq], \
                                            np.array(self.mpc_result.xs)[1, self.pd.nq:], self.pd.r1)
            
            self.q = self.q_interpolated[self.cnt_wbc]
            self.v = self.v_interpolated[self.cnt_wbc]

            # self.result.P = np.array(self.params.Kp_main.tolist() * 4)
            # self.result.D = np.array(self.params.Kd_main.tolist() * 4)
            self.result.FF = self.params.Kff_main * np.ones(12)
            self.result.q_des = self.q
            self.result.v_des = self.v
            self.result.tau_ff = self.mpc_result.us[0] + np.dot(self.mpc_result.K[0], 
                                                                self.mpc.ocp.state.diff(m["x_m"],  self.mpc_result.xs[0]))

            self.xs_init = self.mpc_result.xs[1:] + [self.mpc_result.xs[-1]]
            self.us_init = self.mpc_result.us[1:] + [self.mpc_result.us[-1]]

        t_send = time.time()
        self.t_send = t_send - t_mpc

        #self.clamp_result(device)
        #self.security_check(m)

        if self.error:
            self.set_null_control()

        # self.pyb_camera(device)

        self.t_loop = time.time() - t_start
        self.k += 1
        self.cnt_wbc += 1
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
            elif (np.abs(m["vj_m"]) > 500 * np.pi / 180).any():
                print("-- VELOCITY TOO HIGH ERROR --")
                print(m["vj_m"])
                print(np.abs(m["vj_m"]) > 500 * np.pi / 180)
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

    def read_state(self, device):
        device.parse_sensor_data()
        qj_m = device.joints.positions
        vj_m = device.joints.velocities
        # bp_m = self.tuple_to_array(device.baseState)
        # bv_m = self.tuple_to_array(device.baseVel)
        # if self.pd.useFixedBase == 0:
        #     x_m = np.concatenate([bp_m, qj_m, bv_m, vj_m])
        # else:
        x_m = np.concatenate([qj_m, vj_m])

        return {"qj_m": qj_m, "vj_m": vj_m, "x_m": x_m}

    def interpolate_traj(self, device, q_des, v_des, ratio):
        measures = self.read_state(device)
        qj_des_i = np.linspace(measures["qj_m"], q_des, ratio)
        vj_des_i = np.linspace(measures["vj_m"], v_des, ratio)
        self.interpolated = True

        return qj_des_i, vj_des_i

    def tuple_to_array(self, tup):
        a = np.array([element for tupl in tup for element in tupl])
        return a
