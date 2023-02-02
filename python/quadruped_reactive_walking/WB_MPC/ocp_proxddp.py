"""
Solve the OCP using the PROXDDP package.

Author:
    Wilson Jallet
"""
import time
import numpy as np
import pinocchio as pin
import crocoddyl

import proxddp
from proxddp import manifolds, dynamics

from .problem_data import ProblemData
from .Target import Target
from .ocp_abstract import OCPAbstract
from .ocp_crocoddyl import CrocOCP
from quadruped_reactive_walking import Params


class AlgtrOCP(CrocOCP):
    """Solve the OCP using proxddp."""

    def __init__(
        self,
        pd: ProblemData,
        params: Params,
        footsteps,
        base_refs,
        use_prox=False,
        run_croc=False,
    ):
        super().__init__(pd, params, footsteps, base_refs)

        self.my_problem: proxddp.TrajOptProblem = proxddp.croc.convertCrocoddylProblem(
            self.problem
        )

        self.run_croc_compare = run_croc

        self.verbose = proxddp.VerboseLevel.QUIET
        if params.verbose:
            self.verbose = proxddp.VerboseLevel.VERBOSE
            self.ddp.setCallbacks([crocoddyl.CallbackVerbose()])
        self.tol = 1e-3
        self.mu_init = 1e-5
        # self.prox_ddp = proxddp.SolverProxDDP(self.tol, self.mu_init)
        self.prox_ddp = proxddp.SolverFDDP(self.tol)
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

    def solve(self, k, x0, footstep, base_ref, xs_init=None, us_init=None):
        t_start = time.time()
        self.x0 = x0
        self.make_ocp(k, footstep, base_ref)
        self.my_problem.x0_init = self.x0

        t_update = time.time()
        self.t_update = t_update - t_start

        if xs_init is None or us_init is None:
            nsteps = self.my_problem.num_steps
            xs = [x0] * (nsteps + 1)
            us = self.problem.quasiStatic([x0] * nsteps)
        else:
            xs = xs_init
            us = us_init

        t_warm_start = time.time()
        self.t_warm_start = t_warm_start - t_update

        self.prox_ddp.max_iters = self.max_iter
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

        if self.run_croc_compare:
            # run crocoddyl
            self.ddp.th_stop = prox_norm_2
            self.ddp.solve(xs, us, self.max_iter, False)

            croc_norm_inf = max([infNorm(q) for q in Qus])
            croc_norm_2 = sum([q.dot(q) for q in Qus])

            self.croc_stops.append(croc_norm_inf)
            self.croc_stops_2.append(croc_norm_2)
            self.croc_iters.append(self.ddp.iter)

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
        nsteps = self.problem.T
        feedbacks = [-K.copy() for K in res.controlFeedbacks()]
        xs = res.xs.tolist()
        us = res.us.tolist()

        if self.params.LOGGING and self.run_croc_compare:
            np.set_printoptions(precision=4, linewidth=250)
            X_err = [
                infNorm(self.state.diff(xs[i], self.ddp.xs[i]))
                for i in range(nsteps + 1)
            ]
            U_err = [infNorm(us[i] - self.ddp.us[i]) for i in range(nsteps)]
            self.x_solver_errs.append(max(X_err))
            self.u_solver_errs.append(max(U_err))

            croc_fb = self.ddp.K
            fb_err = [infNorm(feedbacks[i] - croc_fb[i]) for i in range(nsteps)]
            self.fb_errs.append(max(fb_err))
            # if max(self.x_solver_errs[-1], self.u_solver_errs[-1]) > 1e-7:
            #     import ipdb;
            #     ipdb.set_trace()

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
