import pinocchio as pin
import numpy as np


def get_translation(model, x, idx, ref_frame=pin.WORLD):
    q = x[: model.nq]
    v = x[model.nq :]
    rdata = model.createData()
    pin.forwardKinematics(model, rdata, q, v)
    pin.updateFramePlacements(model, rdata)
    frame_p = rdata.oMf[idx].translation
    frame_v = pin.getFrameVelocity(model, rdata, idx, ref_frame).linear
    return frame_p, frame_v


def get_translation_array(model, xs, idx, ref_frame=pin.WORLD, x0=None):
    frame_p = []
    frame_v = []
    rdata = model.createData()
    if isinstance(x0, np.ndarray):
        xiter = np.concatenate([x0.reshape(1, -1), xs])
    else:
        xiter = xs

    for x in xiter:
        q = x[: model.nq]
        v = x[model.nq :]
        pin.forwardKinematics(model, rdata, q, v)
        pin.updateFramePlacements(model, rdata)
        frame_p += [rdata.oMf[idx].translation.copy()]
        frame_v += [pin.getFrameVelocity(model, rdata, idx, ref_frame).linear.copy()]
    return np.array(frame_p), np.array(frame_v)
