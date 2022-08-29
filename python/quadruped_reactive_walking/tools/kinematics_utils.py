import pinocchio as pin
import numpy as np
from ..WB_MPC.ProblemData import ProblemData

def get_translation(pd:ProblemData, x, idx, ref_frame=pin.WORLD):
    q = x[: pd.nq]
    v = x[pd.nq :]
    pin.forwardKinematics(pd.model, pd.rdata, q, v)
    pin.updateFramePlacements(pd.model, pd.rdata)
    frame_p = pd.rdata.oMf[idx].translation
    frame_v = pin.getFrameVelocity(pd.model, pd.rdata, idx, ref_frame).linear
    return frame_p, frame_v

def get_translation_array(pd:ProblemData, x, idx, ref_frame=pin.WORLD, x0=None):
    frame_p = []
    frame_v = []
    if isinstance(x0, np.ndarray):
        xiter = np.concatenate([x0.reshape(1,-1), x])
    else:
        xiter = x

    for xs in xiter:
        q = xs[:pd.nq]
        v = xs[pd.nq:]
        print(v)
        pin.forwardKinematics(pd.model, pd.rdata, q, v)
        pin.updateFramePlacements(pd.model, pd.rdata)
        frame_p += [pd.rdata.oMf[idx].translation.copy()]
        frame_v += [pin.getFrameVelocity(pd.model, pd.rdata, idx, ref_frame).linear.copy()]
    return np.array(frame_p), np.array(frame_v)
