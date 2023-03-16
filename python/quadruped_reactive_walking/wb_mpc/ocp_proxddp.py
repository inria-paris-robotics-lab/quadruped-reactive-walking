"""
Solve the OCP using the PROXDDP package.

Author:
    Wilson Jallet
"""
import time
import crocoddyl

import proxddp
from colorama import Fore

from .ocp_crocoddyl import CrocOCP
from quadruped_reactive_walking import Params
import abc


class AlgtrOCPAbstract(CrocOCP):
    """Solve the OCP using proxddp."""

    # Must be set by child class
    prox_ddp = None  # Solver instance

    @abc.abstractclassmethod
    def get_type_str():
        pass

    def __init__(
        self,
        params: Params,
        footsteps,
        base_refs,
    ):
        super().__init__(params, footsteps, base_refs)

        self.my_problem: proxddp.TrajOptProblem = proxddp.croc.convertCrocoddylProblem(
            self.problem
        )

        self.num_threads = params.ocp.num_threads
        if hasattr(self.problem, "num_threads"):
            self.problem.num_threads = self.num_threads
        self.my_problem.setNumThreads(self.num_threads)

        self.verbose = proxddp.QUIET
        if params.ocp.verbose:
            self.verbose = proxddp.VERBOSE
            self.ddp.setCallbacks([crocoddyl.CallbackVerbose()])

        self.prox_ddp.verbose = self.verbose
        self.prox_ddp.max_iters = self.max_iter
        self.prox_ddp.setup(self.my_problem)

        self.x_solver_errs = []
        self.u_solver_errs = []
        self.fb_errs = []
        self.prox_stops_2 = []
        self.croc_stops_2 = []
        self.prox_stops = []
        self.croc_stops = []
        self.prox_iters = []
        self.croc_iters = []

    def solve(self, k, xs_init=None, us_init=None):
        t_start = time.time()
        self.my_problem.x0_init = self.x0

        t_update = time.time()
        self.t_update = t_update - t_start

        if xs_init is None or us_init is None:
            nsteps = self.my_problem.num_steps
            xs = [self.x0] * (nsteps + 1)
            us = self.problem.quasiStatic([self.x0] * nsteps)
        else:
            xs = xs_init
            us = us_init

        t_warm_start = time.time()
        self.t_warm_start = t_warm_start - t_update

        maxiter = self.max_iter if k > 0 else self.init_max_iters
        self.prox_ddp.max_iters = maxiter
        self.prox_ddp.run(self.my_problem, xs, us)

        # compute proxddp's criteria
        ws = self.prox_ddp.getWorkspace()
        if hasattr(ws, "Qus_ddp"):
            Qus = ws.Qus_ddp
        else:
            Qus = [q.Qu for q in ws.q_params]
        prox_norm_2 = sum(q.dot(q) for q in Qus)

        res = self.prox_ddp.getResults()
        prox_norm_inf = max(res.primal_infeas, res.dual_infeas)
        self.prox_stops.append(prox_norm_inf)
        self.prox_stops_2.append(prox_norm_2)
        self.prox_iters.append(res.num_iters)

        t_ddp = time.time()
        self.t_ddp = t_ddp - t_warm_start

        self.t_solve = time.time() - t_start
        self.num_iters = res.num_iters

    def circular_append(self, action_model: crocoddyl.ActionModelAbstract):
        d = action_model.createData()
        self.problem.circularAppend(action_model, d)

        sm = proxddp.croc.ActionModelWrapper(action_model)
        self.my_problem.replaceStageCircular(sm)
        ws = self.prox_ddp.getWorkspace()
        ws.cycle_append(sm)

    def get_results(self):
        res = self.prox_ddp.getResults()
        ws = self.prox_ddp.getWorkspace()  # noqa
        feedbacks = [-K.copy() for K in res.controlFeedbacks()]
        xs = res.xs.tolist()
        us = res.us.tolist()

        return (
            self.current_gait.copy(),
            xs,
            us,
            feedbacks,
            self.t_ddp,
        )

    def clear(self):
        self.x_solver_errs.clear()
        self.u_solver_errs.clear()
        self.fb_errs.clear()

        self.prox_stops.clear()
        self.prox_stops_2.clear()
        self.prox_iters.clear()

        self.croc_stops.clear()
        self.croc_stops_2.clear()
        self.croc_iters.clear()


class AlgtrOCPFDDP(AlgtrOCPAbstract):
    """Solve the OCP using fddp."""

    def __init__(
        self,
        params: Params,
        footsteps,
        base_refs,
    ):
        print(Fore.BLUE + "[using SolverFDDP]")
        tolerance = 1e-3  # todo: move to params
        self.prox_ddp = proxddp.SolverFDDP(tolerance)
        print(Fore.RESET)
        super().__init__(params, footsteps, base_refs)

    def get_type_str():
        return "algtr-fddp"


class AlgtrOCPProx(AlgtrOCPAbstract):
    """Solve the OCP using proxddp."""

    def __init__(
        self,
        params: Params,
        footsteps,
        base_refs,
    ):
        print(Fore.GREEN + "[using SolverProxDDP]")
        mu_init = 1e-9
        tolerance = 1e-3  # todo: move to params
        self.prox_ddp = proxddp.SolverProxDDP(tolerance, mu_init, 0.0)
        self.prox_ddp.reg_init = 1e-8
        print(Fore.RESET)
        super().__init__(params, footsteps, base_refs)

    def get_type_str():
        return "algtr-prox"
