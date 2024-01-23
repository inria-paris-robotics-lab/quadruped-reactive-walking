"""
Solve the OCP using the PROXDDP package.

Author:
    Wilson Jallet
"""
import time
import aligator
import crocoddyl
import numpy as np

from abc import abstractclassmethod
from colorama import Fore
from .ocp_crocoddyl import CrocOCP
from quadruped_reactive_walking import Params


def infNorm(x):
    return np.linalg.norm(x, ord=np.inf)


class AlgtrOCPAbstract(CrocOCP):
    """Solve the OCP using aligator."""

    # Must be set by child class
    prox_ddp = None  # Solver instance

    @abstractclassmethod
    def get_type_str():
        pass

    def __init__(
        self,
        params: Params,
        footsteps,
        base_refs,
    ):
        super().__init__(params, footsteps, base_refs)

        self.my_problem: aligator.TrajOptProblem = (
            aligator.croc.convertCrocoddylProblem(self.problem)
        )

        self.num_threads = params.ocp.num_threads
        if hasattr(self.problem, "num_threads"):
            self.problem.num_threads = self.num_threads
        self.my_problem.setNumThreads(self.num_threads)

        self.verbose = aligator.QUIET
        if params.ocp.verbose:
            self.verbose = aligator.VERBOSE

        self.prox_ddp.verbose = self.verbose
        self.prox_ddp.max_iters = self.max_iter
        self.prox_ddp.setup(self.my_problem)

    def solve(self, k):
        t_start = time.time()
        self.my_problem.x0_init = self.x0

        t_update = time.time()
        self.t_update = t_update - t_start

        self._check_ws_dim()

        t_warm_start = time.time()
        self.t_warm_start = t_warm_start - t_update

        maxiter = self.max_iter if k > 0 else self.init_max_iters
        self.prox_ddp.max_iters = maxiter
        self.prox_ddp.run(self.my_problem, self.xs_init, self.us_init)

        # compute aligator's criteria
        res = self.prox_ddp.results

        t_ddp = time.time()
        self.t_ddp = t_ddp - t_warm_start

        self.t_solve = time.time() - t_start
        self.num_iters = res.num_iters

    def circular_append(self, action_model: crocoddyl.ActionModelAbstract):
        d = action_model.createData()
        self.problem.circularAppend(action_model, d)

        sm = aligator.croc.ActionModelWrapper(action_model)
        self.my_problem.replaceStageCircular(sm)
        ws = self.prox_ddp.workspace
        ws.cycleAppend(sm.createData())

    def get_results(self, window_size=None):
        res = self.prox_ddp.results
        self.xs_init = res.xs
        self.us_init = res.us
        if window_size is None:
            window_size = len(res.us)
        feedbacks = res.controlFeedbacks()[:window_size]
        # flip sign because controller expects Crocoddyl's convention
        feedbacks = [-K.copy() for K in feedbacks]

        return (
            self.current_gait.copy(),
            res.xs[: window_size + 1],
            res.us[:window_size],
            feedbacks,
            self.t_ddp,
        )


class AlgtrOCPFDDP(AlgtrOCPAbstract):
    """Solve the OCP using fddp."""

    def __init__(
        self,
        params: Params,
        footsteps,
        base_refs,
    ):
        print(Fore.BLUE + "[using SolverFDDP]" + Fore.RESET)
        self.prox_ddp = aligator.SolverFDDP(params.ocp.tol)
        super().__init__(params, footsteps, base_refs)

    def get_type_str():
        return "algtr-fddp"


class AlgtrOCPProx(AlgtrOCPAbstract):
    """Solve the OCP using aligator."""

    def __init__(
        self,
        params: Params,
        footsteps,
        base_refs,
    ):
        print(Fore.GREEN + "[using SolverProxDDP]" + Fore.RESET)
        mu_init = 1e-10
        self.prox_ddp = aligator.SolverProxDDP(params.ocp.tol, mu_init, 0.0)
        self.prox_ddp.mu_min = 1e-12
        self.prox_ddp.reg_init = 1e-9
        self.prox_ddp.ldlt_algo_choice = aligator.LDLT_DENSE
        self.prox_ddp.max_refinement_steps = 0
        super().__init__(params, footsteps, base_refs)

    def get_type_str():
        return "algtr-prox"
