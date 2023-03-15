from example_robot_data import load
import numpy as np
import pinocchio as pin

def make_footstep(q_init):
    # Load robot model and data
    solo = load("solo12")
    q = solo.q0.reshape((-1, 1))
    q[7:, 0] = q_init

    pin.framesForwardKinematics(solo.model, solo.data, q)

    # Initialisation of model quantities
    pin.updateFramePlacements(solo.model, solo.data)

    LEGS = ["FL", "FR", "HL", "HR"]

    # Initialisation of the position of footsteps
    initial_footsteps = np.zeros((3, 4))
    h_init = 0.0
    indexes = [solo.model.getFrameId(leg + "_FOOT") for leg in LEGS]
    for i in range(4):
        initial_footsteps[:, i] = solo.data.oMf[indexes[i]].translation
        h = (solo.data.oMf[1].translation - solo.data.oMf[indexes[i]].translation)[2]
        if h > h_init:
            h_init = h
    initial_footsteps[2, :] = 0.0

    return initial_footsteps


def quaternionToRPY(quat):
    """
    Quaternion (4 x 0) to Roll Pitch Yaw (3 x 1)
    """
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]

    rotateXa0 = 2.0 * (qy * qz + qw * qx)
    rotateXa1 = qw * qw - qx * qx - qy * qy + qz * qz
    rotateX = 0.0

    if (rotateXa0 != 0.0) and (rotateXa1 != 0.0):
        rotateX = np.arctan2(rotateXa0, rotateXa1)

    rotateYa0 = -2.0 * (qx * qz - qw * qy)
    rotateY = 0.0
    if rotateYa0 >= 1.0:
        rotateY = np.pi / 2.0
    elif rotateYa0 <= -1.0:
        rotateY = -np.pi / 2.0
    else:
        rotateY = np.arcsin(rotateYa0)

    rotateZa0 = 2.0 * (qx * qy + qw * qz)
    rotateZa1 = qw * qw + qx * qx - qy * qy - qz * qz
    rotateZ = 0.0
    if (rotateZa0 != 0.0) and (rotateZa1 != 0.0):
        rotateZ = np.arctan2(rotateZa0, rotateZa1)

    return np.array([[rotateX], [rotateY], [rotateZ]])
