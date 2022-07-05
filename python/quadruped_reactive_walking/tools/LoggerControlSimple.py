"""This class will log 1d array in Nd matrix from device and qualisys object"""
from datetime import datetime
from time import time
from pathlib import Path

import numpy as np
import pinocchio as pin


class LoggerControl:
    def __init__(self, params, log_size=60e3, loop_buffer=False):
        self.log_size = np.int(log_size)
        self.i = 0
        self.dt = params.dt_wbc
        self.loop_buffer = loop_buffer

    def intialize(self):
        size = self.log_size

        # TODO: ADD WHAT YOU WANT TO LOG

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

        # Controller timings: MPC time, ...
        self.t_MPC = np.zeros(size)

        # MPC
        # self.mpc_input = np.zeros([size, mpc_result_size])
        # self.mpc_result = np.zeros([size, mpc_result_size])
        self.mpc_solving_duration = np.zeros([size])
        self.mpc_cost = np.zeros([size, 1])

        # Whole body control
        self.wbc_P = np.zeros([size, 12])  # proportionnal gains of the PD+
        self.wbc_D = np.zeros([size, 12])  # derivative gains of the PD+
        self.wbc_q_des = np.zeros([size, 12])  # desired position of actuators
        self.wbc_v_des = np.zeros([size, 12])  # desired velocity of actuators
        self.wbc_FF = np.zeros([size, 12])  # gains for the feedforward torques
        self.wbc_tau_ff = np.zeros([size, 12])  # feedforward torques

    def sample(self, controller, device, qualisys=None):
        if self.i >= self.size:
            if self.loop_buffer:
                self.i = 0
            else:
                return

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

        # Logging from model predictive control
        # TODO update
        # self.mpc_input[self.i] = controller.mpc_result
        # self.mpc_result[self.i] = controller.mpc_result
        self.mpc_solving_duration[self.i] = controller.mpc.t_mpc_solving_duration
        self.mpc_cost[self.i] = controller.mpc_cost

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

    def plotTimes(self):
        """
        Estimated computation time for each step of the control architecture
        """
        from matplotlib import pyplot as plt

        t_range = np.array([k * self.dt for k in range(self.tstamps.shape[0])])

        plt.figure()
        plt.plot(t_range, self.t_MPC, "r+")
        legend = ["MPC"]
        plt.legend(legend)
        plt.xlabel("Time [s]")
        plt.ylabel("Time [s]")
        self.custom_suptitle("Computation time of each block")

    def plotMPCCost(self):
        """
        Plot the cost of the OSQP MPC
        """
        from matplotlib import pyplot as plt

        t_range = np.array([k * self.dt for k in range(self.tstamps.shape[0])])

        fig = plt.figure()
        plt.plot(t_range[100:], self.mpc_cost[100:], "k+")
        plt.legend(["MPC cost"])
        plt.xlabel("Time [s]")
        plt.ylabel("Cost value")
        self.custom_suptitle("MPC cost value")

    def plotMpcTime(self):
        """
        Plot estimated solving time of the model prediction control
        """
        from matplotlib import pyplot as plt

        t_range = np.array([k * self.dt for k in range(self.tstamps.shape[0])])

        fig = plt.figure()
        plt.plot(t_range[35:], self.mpc_solving_duration[35:], "k+")
        plt.legend(["Solving duration"])
        plt.xlabel("Time [s]")
        plt.ylabel("Time [s]")
        self.custom_suptitle("MPC solving time")

    def plot(self, self):
        """ "
        Step in system time at each loop
        """

        from matplotlib import pyplot as plt

        N = self.tstamps.shape[0]
        t_range = np.array([k * self.dt for k in range(N)])

        # TODO add the plots you want
        self.plotTimes()
        self.plotMpcTime()
        self.plotMPCCost()

        plt.show(block=True)

    def save(self, fileName="data"):
        date_str = datetime.now().strftime("_%Y_%m_%d_%H_%M")
        name = fileName + date_str + "_" + str(self.type_MPC) + ".npz"

        np.savez_compressed(
            name,
            t_MPC=self.t_MPC,
            # mpc_result=self.mpc_result,
            mpc_solving_duration=self.mpc_solving_duration,
            mpc_cost=self.mpc_cost,
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
        print("Log saved in " + name)

    def load(self):

        if self.data is None:
            print("No data file loaded. Need one in the constructor.")
            return

        # TODO: update loader

        self.t_MPC = self.data["t_MPC"]
        # self.mpc_x_f = self.data["mpc_x_f"]
        self.mpc_solving_duration = self.data["mpc_solving_duration"]

        self.wbc_P = self.data["wbc_P"]
        self.wbc_D = self.data["wbc_D"]
        self.wbc_q_des = self.data["wbc_q_des"]
        self.wbc_v_des = self.data["wbc_v_des"]
        self.wbc_FF = self.data["wbc_FF"]
        self.wbc_tau_ff = self.data["wbc_tau_ff"]

        self.mpc_cost = self.data["mpc_cost"]
        self.tstamps = self.data["tstamps"]


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

    params = qrw.Params()
    logger = LoggerControl(params)
    logger.load(fileName=args.file)
    logger.plot()
