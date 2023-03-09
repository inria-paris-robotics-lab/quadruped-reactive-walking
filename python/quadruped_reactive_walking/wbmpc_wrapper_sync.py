import numpy as np

from .wb_mpc.ocp_abstract import OCPAbstract
from .wb_mpc.problem_data import TaskSpec

from typing import Type

from .wbmpc_wrapper_abstract import MPCWrapperAbstract, Result

class SyncMPCWrapper(MPCWrapperAbstract):
    """
    Wrapper to run both types of MPC (OQSP or Crocoddyl) in a synchronous manner in the main thread.
    """

    def __init__(
        self, params, footsteps, base_refs, solver_cls: Type[OCPAbstract], **kwargs
    ):
        self.params = params
        self.pd = TaskSpec(params)
        self.T = params.T
        self.nu = self.pd.nu
        self.nx = self.pd.nx
        self.ndx = self.pd.ndx
        self.solver_cls = solver_cls
        self._solver_kwargs = kwargs

        self.ocp = solver_cls(params, footsteps, base_refs, **kwargs)

        self.last_available_result: Result = Result(params)
        self.new_result = False

    def solve(self, k, x0, footstep, base_ref, xs=None, us=None):
        self.ocp.make_ocp(k, x0, footstep, base_ref)
        self.ocp.solve(k, xs, us)
        (
            self.last_available_result.gait,
            self.last_available_result.xs,
            self.last_available_result.us,
            self.last_available_result.K,
            self.last_available_result.solving_duration,
        ) = self.ocp.get_results()
        self.new_result = True

    def get_latest_result(self):
        """
        Return the desired contact forces that have been computed by the last iteration
        of the MPC.
        If a new result is available, return the new result.
        Otherwise return the old result again.
        """
        if self.new_result:
            self.last_available_result.new_result = True
            self.new_result = False
        else:
            self.last_available_result.new_result = False

        return self.last_available_result

    def stop_parallel_loop(self):
        pass # Do nothing since it is single threaded
