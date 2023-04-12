from .wb_mpc.ocp_abstract import OCPAbstract
from quadruped_reactive_walking import MPCResult

from typing import Type

import abc


class MPCWrapperAbstract:
    """
    Wrapper to run both types of MPC (OQSP or Crocoddyl)
    """

    @abc.abstractclassmethod
    def __init__(self, params, footsteps, base_refs, solver_cls: Type[OCPAbstract]):
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


__all__ = ["MPCResult", "MPCWrapperAbstract"]
