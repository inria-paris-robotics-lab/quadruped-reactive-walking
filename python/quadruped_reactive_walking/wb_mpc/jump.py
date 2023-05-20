"""
Construct an OCP for jumping.
"""
# import crocoddyl
# import numpy as np
# import pinocchio as pin
import yaml

from quadruped_reactive_walking import Params
from . import task_spec
from pathlib import Path


def read_jump_yaml():
    fdir = Path.home() / "git-repos/quadruped-reactive-walking/config"
    fp = fdir / "jump_task.yaml"
    with fp.open() as f:
        spec = yaml.safe_load(f)["task"]
        return spec


class JumpOCPBuilder:
    def __init__(self, params: Params):
        self.params = params
        self.task = task_spec.TaskSpec(params)
