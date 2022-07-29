import numpy as np
from .ProblemData import ProblemData
import pinocchio as pin


class Target:
    def __init__(self, pd: ProblemData):
        self.pd = pd
        self.dt = pd.dt

        self.gait = ([] + \
                    [[0, 0, 0, 0]] * pd.init_steps + \
                    [[0, 0, 0, 0]] * pd.target_steps )

        self.T = pd.T
        self.contactSequence = [self.patternToId(p) for p in self.gait]

        self.target = {pd.rfFootId: []}
        q = pd.q0_reduced
        v = pd.v0_reduced
        pin.forwardKinematics(pd.model, pd.rdata, q, v)
        pin.updateFramePlacements(pd.model, pd.rdata)
        self.FR_foot0 = pd.rdata.oMf[pd.rfFootId].translation.copy()
        self.A = np.array([0, 0.03, 0.03])
        self.offset = np.array([0.05, -0.02, 0.06])
        self.freq = np.array([0, 0.5*0, 0.5 * 0])
        self.phase = np.array([0, np.pi / 2, 0])

    def patternToId(self, gait):
        return tuple(self.pd.allContactIds[i] for i, c in enumerate(gait) if c == 1)

    def shift_gait(self):
        self.gait.pop(0)
        self.gait += [self.gait[-1]]
        self.contactSequence = [self.patternToId(p) for p in self.gait]

    def update(self, t):
        target = []
        for n in range(self.T):
            target += [
                self.FR_foot0
                + self.offset
                + self.A
                * np.sin(2 * np.pi * self.freq * (n + t) * self.dt + self.phase)
            ]
        self.target[self.pd.rfFootId] = np.array(target)

    def evaluate_in_t(self, t):
        return {e: self.target[e][t] for e in self.target}
