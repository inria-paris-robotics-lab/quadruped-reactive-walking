"""
Solve the OCP using the PROXDDP package.

Author:
    Wilson Jallet
"""
import time
import numpy as np
import pinocchio as pin

import proxddp
from proxddp import manifolds, dynamics

from .ProblemData import ProblemData
from .Target import Target
from .ocp_abstract import OCPAbstract


class ProxOCP(OCPAbstract):
    def __init__(self, pd: ProblemData, params):
        super().__init__(pd, params)
        self.state = manifolds.MultibodyPhaseSpace(self.rmodel)

    @property
    def rmodel(self):
        return self.pd.model
