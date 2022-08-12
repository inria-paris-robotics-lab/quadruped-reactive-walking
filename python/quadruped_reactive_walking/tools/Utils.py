from example_robot_data import load
import numpy as np
import pinocchio as pin


def init_robot(q_init, params):
    """Load the solo model and initialize the Gepetto viewer if it is enabled

    Args:
        q_init (array): the default position of the robot actuators
        params (object): store parameters
    """
    # Load robot model and data
    solo = load("solo12")
    q = solo.q0.reshape((-1, 1))

    # Initialisation of the position of footsteps to be under the shoulder
    # There is a lateral offset of around 7 centimeters
    fsteps_under_shoulders = np.zeros((3, 4))
    indexes = [
        solo.model.getFrameId("FL_FOOT"),
        solo.model.getFrameId("FR_FOOT"),
        solo.model.getFrameId("HL_FOOT"),
        solo.model.getFrameId("HR_FOOT"),
    ]
    q[7:, 0] = 0.0
    pin.framesForwardKinematics(solo.model, solo.data, q)
    for i in range(4):
        fsteps_under_shoulders[:, i] = solo.data.oMf[indexes[i]].translation
    fsteps_under_shoulders[2, :] = 0.0

    # Initial angular positions of actuators
    q[7:, 0] = q_init

    # Initialisation of model quantities
    pin.centerOfMass(solo.model, solo.data, q, np.zeros((18, 1)))
    pin.updateFramePlacements(solo.model, solo.data)
    pin.crba(solo.model, solo.data, solo.q0)

    # Initialisation of the position of footsteps
    fsteps_init = np.zeros((3, 4))
    h_init = 0.0
    for i in range(4):
        fsteps_init[:, i] = solo.data.oMf[indexes[i]].translation
        h = (solo.data.oMf[1].translation - solo.data.oMf[indexes[i]].translation)[2]
        if h > h_init:
            h_init = h
    fsteps_init[2, :] = 0.0

    # Initialisation of the position of shoulders
    shoulders_init = np.zeros((3, 4))
    indexes = [4, 12, 20, 28]  # Shoulder indexes
    for i in range(4):
        shoulders_init[:, i] = solo.data.oMf[indexes[i]].translation

    # Saving data
    params.h_ref = h_init
    # Mass of the whole urdf model (also = to Ycrb[1].mass)
    params.mass = solo.data.mass[0]
    # Composite rigid body inertia in q_init position
    params.I_mat = solo.data.Ycrb[1].inertia.ravel().tolist()
    params.CoM_offset = (solo.data.com[0][:3] - q[0:3, 0]).tolist()
    params.CoM_offset[1] = 0.0

    #  Use initial feet pos as reference
    for i in range(4):
        for j in range(3):
            params.shoulders[3 * i + j] = shoulders_init[j, i]
            params.footsteps_init[3 * i + j] = fsteps_init[j, i]
            params.footsteps_under_shoulders[3 * i + j] = fsteps_init[j, i]


def quaternionToRPY(quat):
    """Quaternion (4 x 0) to Roll Pitch Yaw (3 x 1)"""

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
