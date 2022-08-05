import numpy as np
from .ProblemData import ProblemData
import pinocchio as pin


class Target:
    def __init__(self, pd: ProblemData):
        self.dt = pd.dt
        pin.forwardKinematics(pd.model, pd.rdata, pd.q0_reduced, pd.v0_reduced)
        pin.updateFramePlacements(pd.model, pd.rdata)
        self.foot_pose = pd.rdata.oMf[pd.rfFootId].translation.copy()
        self.A = np.array([0, 0.03, 0.03])
        self.offset = np.array([0.05, -0.02, 0.06])
        self.freq = np.array([0, 0.5, 0.5])
        self.phase = np.array([0, np.pi / 2, 0])
        self.t_offset = 0

    def shift(self):
        self.t_offset += 1

    def evaluate_circle(self, t):
        return (
            self.foot_pose
            + self.offset
            + self.A
            * np.sin(2 * np.pi * self.freq * (self.t_offset + t) * self.dt + self.phase)
        )

    def footstep(self, t):
        footstep = np.zeros((3, 4))
        footstep[:, 1] = self.evaluate_circle(t)
        return footstep
