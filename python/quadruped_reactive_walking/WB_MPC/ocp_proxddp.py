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


class ProxOCP(CrocOCP):
    """Solve the OCP using proxddp."""

    def __init__(self, pd: ProblemData, params, footsteps, base_refs):
        super().__init__(pd, params, footsteps, base_refs)

        self.rdata = self.rmodel.createData()
        self.my_problem: proxddp.TrajOptProblem = proxddp.croc.convertCrocoddylProblem(
            self.problem
        )
        nsteps = self.my_problem.num_steps
        verbose = proxddp.VerboseLevel.QUIET
        self.verbose = verbose
        self.tol = 1e-3
        self.max_iter = 50
        self.mu_init = 1e-5
        # self.my_ddp = proxddp.SolverProxDDP(
        #     self.tol, self.mu_init, verbose=verbose, max_iters=self.max_iter
        # )
        self.my_ddp = proxddp.SolverFDDP(
            self.tol, verbose=self.verbose, max_iters=self.max_iter
        )
        self.my_ddp.setup(self.my_problem)

        # self.ddp.setCallbacks([crocoddyl.CallbackVerbose()])
        self.ddp.th_stop = self.tol**2.0 * nsteps

        self.x_solver_errs = []
        self.u_solver_errs = []
        self.prox_stops_2 = []
        self.croc_stops_2 = []
        self.prox_stops = []
        self.croc_stops = []
        self.prox_iters = []
        self.croc_iters = []

    def solve(self, k, x0, footstep, base_ref, xs_init=None, us_init=None):
        import pprint

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

        # self.ddp.solve(xs, us, self.max_iter, False)

        # Qus = self.ddp.Qu.tolist().copy()
        # croc_infty_norms_ = max([np.linalg.norm(Q, np.inf) for Q in Qus])
        # print("grad l∞ norm = {:.4e}".format(croc_infty_norms_))
        # self.croc_stops.append(croc_infty_norms_)

        # self.my_ddp.target_tol = croc_infty_norms_
        self.my_ddp.run(self.my_problem, xs, us)
        self.my_ddp.reg_init = self.my_ddp.xreg

        # compute proxddp's criteria
        ws = self.my_ddp.getWorkspace()
        qparams = ws.q_params
        Qus = [qp.Qu for qp in qparams]
        prox_norm_2 = sum(q.dot(q) for q in Qus)

        res = self.my_ddp.getResults()
        prox_norm_inf = max(res.primal_infeas, res.dual_infeas)
        self.prox_stops.append(prox_norm_inf)
        self.prox_stops_2.append(prox_norm_2)
        self.prox_iters.append(res.num_iters)

        # run crocoddyl
        self.ddp.th_stop = prox_norm_2
        self.ddp.solve(xs, us, self.max_iter, False)

        croc_norm_inf = max([np.linalg.norm(q, np.inf) for q in self.ddp.Qu])
        croc_norm_2 = self.ddp.stop

        self.croc_stops.append(croc_norm_inf)
        self.croc_stops_2.append(croc_norm_2)
        self.croc_iters.append(self.ddp.iter + 1)

        t_ddp = time.time()
        self.t_ddp = t_ddp - t_warm_start

        self.t_solve = time.time() - t_start

    def circular_append(self, action_model: crocoddyl.ActionModelAbstract):
        d = action_model.createData()
        self.problem.circularAppend(action_model, d)

        sm = proxddp.croc.ActionModelWrapper(action_model)
        self.my_problem.replaceStageCircular(sm)
        # self.my_ddp.setup(self.my_problem)
        ws = self.my_ddp.getWorkspace()
        ws.cycle_append(sm)

    def get_results(self):
        res = self.my_ddp.getResults()
        nsteps = self.problem.T
        X_err = [np.linalg.norm(res.xs[i] - self.ddp.xs[i]) for i in range(nsteps + 1)]
        U_err = [np.linalg.norm(res.us[i] - self.ddp.us[i]) for i in range(nsteps)]
        self.x_solver_errs.append(max(X_err))
        self.u_solver_errs.append(max(U_err))

        print("X ERR = {:.3e}".format(self.u_solver_errs[-1]), end=" ")
        print("U ERR = {:.3e}".format(self.x_solver_errs[-1]))

        # return super().get_results()

        feedbacks = [-K.copy() for K in res.ctrl_feedbacks]
        return (
            self.current_gait.copy(),
            res.xs.tolist().copy(),
            res.us.tolist().copy(),
            feedbacks,
            self.t_ddp,
        )

    def clear(self):
        self.x_solver_errs.clear()
        self.u_solver_errs.clear()

        self.prox_stops.clear()
        self.prox_stops_2.clear()
        self.prox_iters.clear()

        self.croc_stops.clear()
        self.croc_stops_2.clear()
        self.croc_iters.clear()
