from datetime import datetime
from time import time
import numpy as np
from .kinematics_utils import get_translation, get_translation_array


class LoggerControl:
    def __init__(self, pd, log_size=60e3, loop_buffer=False, file=None):
        if file is not None:
            self.data = np.load(file, allow_pickle=True)

        self.log_size = np.int(log_size)
        self.i = 0
        self.loop_buffer = loop_buffer

        size = self.log_size
        self.pd = pd

        # IMU and actuators:
        self.q_mes = np.zeros([size, 12])
        self.v_mes = np.zeros([size, 12])
        self.torquesFromCurrentMeasurment = np.zeros([size, 12])
        self.baseOrientation = np.zeros([size, 3])
        self.baseOrientationQuat = np.zeros([size, 4])
        self.baseAngularVelocity = np.zeros([size, 3])
        self.baseLinearAcceleration = np.zeros([size, 3])
        self.baseAccelerometer = np.zeros([size, 3])
        self.current = np.zeros(size)
        self.voltage = np.zeros(size)
        self.energy = np.zeros(size)

        # Motion capture:
        self.mocapPosition = np.zeros([size, 3])
        self.mocapVelocity = np.zeros([size, 3])
        self.mocapAngularVelocity = np.zeros([size, 3])
        self.mocapOrientationMat9 = np.zeros([size, 3, 3])
        self.mocapOrientationQuat = np.zeros([size, 4])

        # Timestamps
        self.tstamps = np.zeros(size)

        # TODO: ADD WHAT YOU WANT TO LOG

        # Controller timings: MPC time, ...

        self.ocp_timings = np.zeros([size])
        self.ocp_storage = {
            "xs": np.zeros([size, pd.T + 1, pd.nx]),
            "us": np.zeros([size, pd.T, pd.nu]),
        }

        # MPC

        # Whole body control
        self.wbc_P = np.zeros([size, 12])  # proportionnal gains of the PD+
        self.wbc_D = np.zeros([size, 12])  # derivative gains of the PD+
        self.wbc_q_des = np.zeros([size, 12])  # desired position of actuators
        self.wbc_v_des = np.zeros([size, 12])  # desired velocity of actuators
        self.wbc_FF = np.zeros([size, 12])  # gains for the feedforward torques
        self.wbc_tau_ff = np.zeros([size, 12])  # feedforward torques

    def sample(self, controller, device, qualisys=None):
        # if self.i >= self.size:
        #     if self.loop_buffer:
        #         self.i = 0
        #     else:
        #         return

        # Logging from the device (data coming from the robot)
        self.q_mes[self.i] = device.joints.positions
        self.v_mes[self.i] = device.joints.velocities
        self.baseOrientation[self.i] = device.imu.attitude_euler
        self.baseOrientationQuat[self.i] = device.imu.attitude_quaternion
        self.baseAngularVelocity[self.i] = device.imu.gyroscope
        self.baseLinearAcceleration[self.i] = device.imu.linear_acceleration
        self.baseAccelerometer[self.i] = device.imu.accelerometer
        self.torquesFromCurrentMeasurment[self.i] = device.joints.measured_torques
        self.current[self.i] = device.powerboard.current
        self.voltage[self.i] = device.powerboard.voltage
        self.energy[self.i] = device.powerboard.energy

        # Logging from qualisys (motion capture)
        if qualisys is not None:
            self.mocapPosition[self.i] = qualisys.getPosition()
            self.mocapVelocity[self.i] = qualisys.getVelocity()
            self.mocapAngularVelocity[self.i] = qualisys.getAngularVelocity()
            self.mocapOrientationMat9[self.i] = qualisys.getOrientationMat9()
            self.mocapOrientationQuat[self.i] = qualisys.getOrientationQuat()
        else:  # Logging from PyBullet simulator through fake device
            self.mocapPosition[self.i] = device.baseState[0]
            self.mocapVelocity[self.i] = device.baseVel[0]
            self.mocapAngularVelocity[self.i] = device.baseVel[1]
            self.mocapOrientationMat9[self.i] = device.rot_oMb
            self.mocapOrientationQuat[self.i] = device.baseState[1]

        # Controller timings: MPC time, ...

        # Logging from model predictive control
        self.ocp_timings[self.i] = controller.mpc.ocp.results.solver_time
        self.ocp_storage["xs"][self.i] = np.array(controller.mpc.ocp.results.x)
        self.ocp_storage["us"][self.i] = np.array(controller.mpc.ocp.results.u)

        # Logging from whole body control
        self.wbc_P[self.i] = controller.result.P
        self.wbc_D[self.i] = controller.result.D
        self.wbc_q_des[self.i] = controller.result.q_des
        self.wbc_v_des[self.i] = controller.result.v_des
        self.wbc_FF[self.i] = controller.result.FF
        self.wbc_tau_ff[self.i] = controller.result.tau_ff

        # Logging timestamp
        self.tstamps[self.i] = time()

        self.i += 1

    def plot(self, save=False, fileName="tmp/"):
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("QtAgg")
        plt.style.use("seaborn")

        horizon = self.ocp_storage["xs"].shape[0]
        t15 = np.linspace(0, horizon * self.pd.dt, horizon + 1)
        t1 = np.linspace(0, (horizon) * self.pd.dt, (horizon) * self.pd.r1 + 1)
        t_mpc = np.linspace(0, (horizon) * self.pd.dt, horizon + 1)

        all_ocp_feet_p_log = {
            idx: [
                get_translation_array(self.pd, x, idx)[0]
                for x in self.ocp_storage["xs"]
            ]
            for idx in self.pd.allContactIds
        }
        for foot in all_ocp_feet_p_log:
            all_ocp_feet_p_log[foot] = np.array(all_ocp_feet_p_log[foot])

        legend = ["Hip", "Shoulder", "Knee"]
        plt.figure(figsize=(12, 6), dpi=90)
        plt.title("Solver timings")
        plt.hist(self.ocp_timings, 30)
        plt.xlabel("timee [s]")
        plt.ylabel("Number of cases [#]")
        plt.draw()
        if save:
            plt.savefig(fileName + "_solver_timings")

        legend = ["Hip", "Shoulder", "Knee"]
        plt.figure(figsize=(12, 6), dpi=90)
        i = 0
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title("Joint position of " + str(i))
            [
                plt.plot(np.array(self.q_mes)[:, (3 * i + jj)] * 180 / np.pi)
                for jj in range(3)
            ]
            plt.ylabel("Joint position [deg]")
            plt.xlabel("t[s]")
            plt.legend(legend)
        plt.draw()
        if save:
            plt.savefig(fileName + "_joint_positions")

        legend = ["Hip", "Shoulder", "Knee"]
        plt.figure(figsize=(12, 6), dpi=90)
        i = 0
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title("Joint velocity of " + str(i))
            [
                plt.plot(np.array(self.v_mes)[:, (3 * i + jj)] * 180 / np.pi)
                for jj in range(3)
            ]
            plt.ylabel("Joint velocity [deg/s]")
            plt.xlabel("t[s]")
            plt.legend(legend)
        plt.draw()
        if save:
            plt.savefig(fileName + "_joint_velocities")

        legend = ["Hip", "Shoulder", "Knee"]
        plt.figure(figsize=(12, 6), dpi=90)
        i = 0
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.title("Joint torques of " + str(i))
            [
                plt.plot(np.array(self.torquesFromCurrentMeasurment)[:, (3 * i + jj)])
                for jj in range(3)
            ]
            plt.ylabel("Torque [Nm]")
            plt.xlabel("t[s]")
            plt.legend(legend)
        plt.draw()
        if save:
            plt.savefig(fileName + "_joint_torques")

        """ legend = ['x', 'y', 'z']
        plt.figure(figsize=(12, 18), dpi = 90)
        for p in range(3):
            plt.subplot(3,1, p+1)
            plt.title('Free foot on ' + legend[p])
            for i in range(horizon-1):
                t = np.linspace(i*self.pd.dt, (self.pd.T+ i)*self.pd.dt, self.pd.T+1)
                y = all_ocp_feet_p_log[self.pd.rfFootId][i+1][:,p]
                for j in range(len(y) - 1):
                    plt.plot(t[j:j+2], y[j:j+2], color='royalblue', linewidth = 3, marker='o' ,alpha=max([1 - j/len(y), 0]))
            #plt.plot(t_mpc, feet_p_log_mpc[18][:, p], linewidth=0.8, color = 'tomato', marker='o')
            #plt.plot(t1, feet_p_log_m[18][:, p], linewidth=2, color = 'lightgreen')
        plt.draw() """

        plt.show()

        # TODO add the plots you want

    def save(self, fileName="data"):
        # date_str = datetime.now().strftime("_%Y_%m_%d_%H_%M")
        # name = fileName + date_str + "_" + str(self.type_MPC) + ".npz"

        np.savez_compressed(
            fileName,
            # t_MPC=self.t_MPC,
            ocp_storage=self.ocp_storage,
            mpc_solving_duration=self.ocp_timings,
            # mpc_cost=self.mpc_cost,
            wbc_P=self.wbc_P,
            wbc_D=self.wbc_D,
            wbc_q_des=self.wbc_q_des,
            wbc_v_des=self.wbc_v_des,
            wbc_FF=self.wbc_FF,
            wbc_tau_ff=self.wbc_tau_ff,
            tstamps=self.tstamps,
            q_mes=self.q_mes,
            v_mes=self.v_mes,
            baseOrientation=self.baseOrientation,
            baseOrientationQuat=self.baseOrientationQuat,
            baseAngularVelocity=self.baseAngularVelocity,
            baseLinearAcceleration=self.baseLinearAcceleration,
            baseAccelerometer=self.baseAccelerometer,
            torquesFromCurrentMeasurment=self.torquesFromCurrentMeasurment,
            mocapPosition=self.mocapPosition,
            mocapVelocity=self.mocapVelocity,
            mocapAngularVelocity=self.mocapAngularVelocity,
            mocapOrientationMat9=self.mocapOrientationMat9,
            mocapOrientationQuat=self.mocapOrientationQuat,
            current=self.current,
            voltage=self.voltage,
            energy=self.energy,
        )
        print("Logs and plots saved in " + fileName)

    def load(self):
        if self.data is None:
            print("No data file loaded. Need one in the constructor.")
            return

        # Load sensors arrays
        self.q_mes = self.data["q_mes"]
        self.v_mes = self.data["v_mes"]
        self.baseOrientation = self.data["baseOrientation"]
        self.baseOrientationQuat = self.data["baseOrientationQuat"]
        self.baseAngularVelocity = self.data["baseAngularVelocity"]
        self.baseLinearAcceleration = self.data["baseLinearAcceleration"]
        self.baseAccelerometer = self.data["baseAccelerometer"]
        self.torquesFromCurrentMeasurment = self.data["torquesFromCurrentMeasurment"]

        self.mocapPosition = self.data["mocapPosition"]
        self.mocapVelocity = self.data["mocapVelocity"]
        self.mocapAngularVelocity = self.data["mocapAngularVelocity"]
        self.mocapOrientationMat9 = self.data["mocapOrientationMat9"]
        self.mocapOrientationQuat = self.data["mocapOrientationQuat"]
        self.size = self.q_mes.shape[0]
        self.current = self.data["current"]
        self.voltage = self.data["voltage"]
        self.energy = self.data["energy"]

        # TODO: load your new data
        self.ocp_timings = self.data["mpc_solving_duration"]
        self.ocp_storage = self.data["ocp_storage"].item()

        self.wbc_P = self.data["wbc_P"]
        self.wbc_D = self.data["wbc_D"]
        self.wbc_q_des = self.data["wbc_q_des"]
        self.wbc_v_des = self.data["wbc_v_des"]
        self.wbc_FF = self.data["wbc_FF"]
        self.wbc_tau_ff = self.data["wbc_tau_ff"]

        self.tstamps = self.data["tstamps"]


if __name__ == "__main__":
    import sys
    import os
    import argparse
    import quadruped_reactive_walking as qrw
    from quadruped_reactive_walking.tools import self

    sys.path.insert(0, os.getcwd())

    parser = argparse.ArgumentParser(description="Process logs.")
    parser.add_argument("--file", type=str, help="A valid log file path")
    args = parser.parse_args()

    logger = LoggerControl(file=args.file)
    logger.load()
    logger.plot()
