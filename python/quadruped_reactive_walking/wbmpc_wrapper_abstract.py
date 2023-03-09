import numpy as np

from .wb_mpc.ocp_abstract import OCPAbstract
from .wb_mpc.problem_data import TaskSpec

from typing import Type

import abc

class Result:
    def __init__(self, params):
        pd = TaskSpec(params)
        self.gait = np.zeros((params.T + 1, 4))
        self.xs = list(np.zeros((params.T + 1, pd.nx)))
        self.us = list(np.zeros((params.T, pd.nu)))
        self.K = list(np.zeros([params.T, pd.nu, pd.ndx]))
        self.solving_duration = 0.0
        self.num_iters = 0
        self.new_result = False

class MPCWrapperAbstract:
    """
    Wrapper to run both types of MPC (OQSP or Crocoddyl) with the possibility to run OSQP in
    a parallel process
    """

    @abc.abstractclassmethod
    def __init__(
        self, params, footsteps, base_refs, solver_cls: Type[OCPAbstract], **kwargs
    ):
        pass

    @abc.abstractclassmethod
    def solve(self, k, x0, footstep, base_ref, xs=None, us=None):
        pass

    @abc.abstractclassmethod
    def get_latest_result(self):
        pass

    @abc.abstractclassmethod
    def stop_parallel_loop(self):
        pass