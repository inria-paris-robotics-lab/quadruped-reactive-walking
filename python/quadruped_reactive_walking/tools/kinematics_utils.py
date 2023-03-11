import pinocchio as pin
import numpy as np
from ..wb_mpc.problem_data import TaskSpec


def get_translation(pd: TaskSpec, x, idx, ref_frame=pin.WORLD):
    q = x[: pd.nq]
    v = x[pd.nq :]
    rdata = pd.create_rdata()
    pin.forwardKinematics(pd.model, rdata, q, v)
    pin.updateFramePlacements(pd.model, rdata)
    frame_p = rdata.oMf[idx].translation
    frame_v = pin.getFrameVelocity(pd.model, rdata, idx, ref_frame).linear
    return frame_p, frame_v


def get_translation_array(pd: TaskSpec, x, idx, ref_frame=pin.WORLD, x0=None):
    frame_p = []
    frame_v = []
    rdata = pd.create_rdata()
    if isinstance(x0, np.ndarray):
        xiter = np.concatenate([x0.reshape(1, -1), x])
    else:
        xiter = x

    for xs in xiter:
        q = xs[: pd.nq]
        v = xs[pd.nq :]
        pin.forwardKinematics(pd.model, rdata, q, v)
        pin.updateFramePlacements(pd.model, rdata)
        frame_p += [rdata.oMf[idx].translation.copy()]
        frame_v += [pin.getFrameVelocity(pd.model, rdata, idx, ref_frame).linear.copy()]
    return np.array(frame_p), np.array(frame_v)
