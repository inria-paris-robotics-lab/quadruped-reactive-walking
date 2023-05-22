"""
Construct an OCP for jumping.
"""
import crocoddyl
import numpy as np

# import pinocchio as pin
import yaml

from quadruped_reactive_walking import Params
from pathlib import Path
from . import walking
from ..tools.utils import make_initial_footstep


def read_jump_yaml():
    # fdir = Path.home() / "git-repos/quadruped-reactive-walking/config"
    fdir = Path.cwd() / "config"
    fp = fdir / "jump_task.yaml"
    with fp.open() as f:
        spec = yaml.safe_load(f)["task"]
        return spec


class JumpOCPBuilder:
    def __init__(self, params: Params, footsteps, base_vel_refs):
        self.params = params
        self._base_builder = walking.WalkingOCPBuilder(params, footsteps, base_vel_refs)
        self.task = self._base_builder.task
        self.state = self._base_builder.state
        self.rdata = self._base_builder.rdata

        self.x0 = self.task.x0
        self.jump_spec = read_jump_yaml()
        feet_pos = make_initial_footstep(params.q_init)
        self.ground_models_1 = self.create_ground_models(feet_pos)
        self.jump_models = self.create_jump_model()
        self.landing_model = None
        self.problem = crocoddyl.ShootingProblem(self.x0, *self.ground_models_1)

    def create_ground_models(self, feet_pos):
        rms = []
        support_feet = np.asarray(self.task.feet_ids)
        for k in range(self.params.N_gait):
            m = self._base_builder.make_running_model(support_feet, [], feet_pos, None)
            rms.append(m)
        return rms, self._base_builder.make_terminal_model(support_feet)

    def create_jump_model(self):
        support_feet = np.array([])
        m = self._base_builder._create_standard_model(support_feet)
        return m
