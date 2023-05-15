from .wb_mpc.task_spec import TaskSpec
from quadruped_reactive_walking import MPCResult, IMPCWrapper

import abc


class _MPCMeta(type(IMPCWrapper), abc.ABCMeta):
    pass


class MPCWrapperAbstract(IMPCWrapper, metaclass=_MPCMeta):
    """
    Wrapper to run both types of MPC (OQSP or Crocoddyl)
    """

    def __init__(self, params):
        super().__init__(params)
        self.pd = TaskSpec(params)

    @property
    def nx(self):
        return self.pd.nx

    @property
    def ndx(self):
        return self.pd.ndx

    @property
    def nu(self):
        return self.pd.nu

    @abc.abstractclassmethod
    def stop_parallel_loop(self):
        pass


__all__ = ["MPCResult", "MPCWrapperAbstract"]
