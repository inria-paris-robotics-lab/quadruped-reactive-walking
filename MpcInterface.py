# coding: utf8

import numpy as np
import pybullet as pyb
import pinocchio as pin


class MpcInterface:

    def __init__(self):

        # Initialisation of matrices
        self.oMb = pin.SE3.Identity()  # transform from world to base frame ("1")
        self.oMl = pin.SE3.Identity()  #  transform from world to local frame ("L")
        self.RPY = np.zeros((3, 1))  # roll, pitch, yaw of the base in world frame
        self.oC = np.zeros((3, ))  #  position of the CoM in world frame
        self.oV = np.zeros((3, ))  #  linear velocity of the CoM in world frame
        self.oW = np.zeros((3, 1))  # angular velocity of the CoM in world frame
        self.lC = np.zeros((3, ))  #  position of the CoM in local frame
        self.lV = np.zeros((3, ))  #  linear velocity of the CoM in local frame
        self.lW = np.zeros((3, 1))  #  angular velocity of the CoM in local frame
        self.lRb = np.eye(3)  # rotation matrix from the local frame to the base frame
        self.abg = np.zeros((3, 1))  # roll, pitch, yaw of the base in local frame
        self.l_feet = np.zeros((3, 4))  # position of feet in local frame

        # Indexes of feet frames
        self.indexes = [10, 18, 26, 34]

        # Average height of feet in local frame
        self.mean_feet_z = 0.0

    def update(self, solo, qmes12, vmes12):

        ################
        # Process data #
        ################

        # Get center of mass from Pinocchio
        pin.centerOfMass(solo.model, solo.data, qmes12, vmes12)

        # Update position/orientation of frames
        pin.updateFramePlacements(solo.model, solo.data)

        # Update average height of feet
        self.mean_feet_z = 0.0
        """for i in self.indexes:
            self.mean_feet_z += solo.data.oMf[i].translation[2, 0]
        self.mean_feet_z *= 0.25"""
        for i in self.indexes:
            self.mean_feet_z = np.min((self.mean_feet_z, solo.data.oMf[i].translation[2, 0]))

        # Store position, linear velocity and angular velocity in global frame
        self.oC = solo.data.com[0]
        self.oV = solo.data.vcom[0]
        self.oW = vmes12[3:6]

        # Get SE3 object from world frame to base frame
        self.oMb = pin.SE3(pin.Quaternion(qmes12[3:7]), self.oC)
        self.RPY = pin.rpy.matrixToRpy(self.oMb.rotation)

        # Get SE3 object from world frame to local frame
        self.oMl = pin.SE3(pin.utils.rotate('z', self.RPY[2, 0]),
                           np.array([self.oC[0, 0], self.oC[1, 0], self.mean_feet_z]))

        # Get position, linear velocity and angular velocity in local frame
        self.lC = self.oMl.inverse() * self.oC
        self.lV = self.oMl.rotation.transpose() @ self.oV
        self.lW = self.oMl.rotation.transpose() @ self.oW

        # Position of feet in local frame
        for i, j in enumerate(self.indexes):
            self.l_feet[:, i:(i+1)] = self.oMl.inverse() * solo.data.oMf[j].translation

        # Orientation of the base in local frame
        # Base and local frames have the same yaw orientation in world frame
        self.abg[0:2] = self.RPY[0:2]

        return 0
