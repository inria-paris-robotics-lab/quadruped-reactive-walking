from tracemalloc import start
from xxlimited import foo

from .ProblemData import ProblemData
from .Target import Target
import crocoddyl
import pinocchio as pin
import numpy as np
from time import time


class OCP:
    def __init__(self, pd: ProblemData, footsteps, gait):
        self.pd = pd
        self.max_iter = 1

        self.state = crocoddyl.StateMultibody(self.pd.model)
        self.initialized = False
        self.t_problem_update = 0
        self.t_update_last_model = 0.0
        self.t_shift = 0.0

        rm, tm = self.initialize_models(gait, footsteps)

        self.x0 = self.pd.x0_reduced

        self.problem = crocoddyl.ShootingProblem(
            self.x0, rm, tm)
        self.ddp = crocoddyl.SolverFDDP(self.problem)

    def initialize_models(self, gait, footsteps):
        models = []
        for t, step in enumerate(gait):
            tasks = self.make_task(footsteps[t])
            support_feet = [self.pd.allContactIds[i] for i in np.nonzero(step == 1)[0]]
            models.append(self.create_model(support_feet, tasks))

        support_feet = [self.pd.allContactIds[i] for i in np.nonzero(gait[-1] == 1)[0]]
        terminal_model = self.create_model(support_feet, is_terminal=True)

        return models, terminal_model

    def solve(self, x0, footstep, gait, xs_init=None, us_init=None):
        t_start = time()
        self.x0 = x0
        self.make_ocp(footstep, gait)

        t_update = time()
        self.t_update = t_update - t_start

        if xs_init is None or us_init is None:
            xs = [x0] * (self.ddp.problem.T + 1)
            us = self.ddp.problem.quasiStatic([x0] * self.ddp.problem.T)
        else:
            xs = xs_init
            us = us_init

        t_warm_start = time()
        self.t_warm_start = t_warm_start - t_update

        # self.ddp.setCallbacks([crocoddyl.CallbackVerbose()])
        self.ddp.solve(xs, us, self.max_iter, False)

        t_ddp = time()
        self.t_ddp = t_ddp - t_warm_start

        self.t_solve = time() - t_start

    def make_ocp(self, footstep, gait):
        """
        Create a shooting problem for a simple walking gait.

        :param x0: initial state
        """
        pin.forwardKinematics(self.pd.model, self.pd.rdata, self.x0[: self.pd.nq])
        pin.updateFramePlacements(self.pd.model, self.pd.rdata)

        if self.initialized:
            tasks = self.make_task(footstep)
            support_feet = [
                self.pd.allContactIds[i] for i in np.nonzero(gait[-1] == 1)[0]
            ]
            self.update_model(self.problem.runningModels[0], tasks, support_feet)

            self.problem.circularAppend(
                self.problem.runningModels[0],
                self.problem.runningModels[0].createData(),
            )

        self.problem.x0 = self.x0

        self.initialized = True

    def make_task(self, footstep):
        task = [[], []]
        for foot in range(4):
            if footstep[:, foot].any():
                task[0].append(self.pd.allContactIds[foot])
                task[1].append(footstep[:, foot])
        return task

    def get_results(self):
        return (
            self.ddp.xs.tolist().copy(),
            self.ddp.us.tolist().copy(),
            self.ddp.K.tolist().copy(),
            self.t_ddp,
        )

    def get_croco_forces(self):
        d = self.ddp.problem.runningDatas[0]
        cnames = d.differential.multibody.contacts.contacts.todict().keys()
        forces = {n: [] for n in cnames}

        for m in self.ddp.problem.runningDatas:
            mdict = m.differential.multibody.contacts.contacts.todict()
            for n in cnames:
                if n in mdict:
                    forces[n] += [(mdict[n].jMf.inverse() * mdict[n].f).linear]
                else:
                    forces[n] += [np.array([0, 0, 0])]
        for f in forces:
            forces[f] = np.array(forces[f])
        return forces

    def get_croco_forces_ws(self):
        forces = []

        for m in self.ddp.problem.runningDatas:
            mdict = m.differential.multibody.contacts.contacts.todict()
            f_tmp = []
            for n in mdict:
                f_tmp += [(mdict[n].jMf.inverse() * mdict[n].f).linear]
            forces += [np.concatenate(f_tmp)]
        return forces

    def get_croco_acc(self):
        acc = []
        [acc.append(m.differential.xout) for m in self.ddp.problem.runningDatas]
        return acc

    def update_model(self, model, tasks, support_feet):
        for i in self.pd.allContactIds:
            name = self.pd.model.frames[i].name + "_contact"
            model.differential.contacts.changeContactStatus(name, i in support_feet)

        self.update_tracking_costs(model.differential.costs, tasks)


    def create_model(self, support_feet=[], tasks=[], is_terminal=False):
        """
        Create the action model

        :param state: swinging foot task
        :param support_feet: list of support feet ids
        :param task: list of support feet ids and associated tracking reference
        :param isTterminal: true for the terminal node
        :return action model for a swing foot phase
        """
        model = self.create_standard_model(support_feet)
        if is_terminal:
            self.make_terminal_model(model)
        else:
            self.make_running_model(model, support_feet, tasks)

        return model

    def create_standard_model(self, support_feet):
        """
        Create a standard action model

        :param state: swinging foot task
        :param support_feet: list of support feet ids
        :return action model for a swing foot phase
        """
        if self.pd.useFixedBase == 0:
            actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        else:
            actuation = crocoddyl.ActuationModelFull(self.state)
        nu = actuation.nu

        control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        contacts = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in self.pd.allContactIds:
            name = self.pd.model.frames[i].name + "_contact"
            contact = crocoddyl.ContactModel3D(
                self.state, i, np.zeros(3), nu, np.zeros(2)
            )
            contacts.addContact(name, contact)
            if i not in support_feet:
                contacts.changeContactStatus(name, False)

        costs = crocoddyl.CostModelSum(self.state, nu)
        residual = crocoddyl.ResidualModelState(self.state, self.pd.xref, nu)
        activation = crocoddyl.ActivationModelWeightedQuad(self.pd.state_reg_w**2)
        state_cost = crocoddyl.CostModelResidual(self.state, activation, residual)
        costs.addCost("state_reg", state_cost, 1)

        differential = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, actuation, contacts, costs, 0.0, True
        )
        model = crocoddyl.IntegratedActionModelEuler(differential, control, self.pd.dt)

        return model

    def make_terminal_model(self, model):
        """
        Add the final velocity cost to the terminal model
        """
        nu = model.differential.actuation.nu
        residual = crocoddyl.ResidualModelState(self.state, self.pd.xref, nu)
        activation = crocoddyl.ActivationModelWeightedQuad(
            self.pd.terminal_velocity_w**2
        )
        state_cost = crocoddyl.CostModelResidual(self.state, activation, residual)
        model.differential.costs.addCost("terminal_velocity", state_cost, 1)

    def make_running_model(self, model, support_feet, tasks):
        """
        Add all the costs to the running models
        """
        nu = model.differential.actuation.nu
        costs = model.differential.costs
        for i in self.pd.allContactIds:
            cone = crocoddyl.FrictionCone(self.pd.Rsurf, self.pd.mu, 4, False)
            residual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, i, cone, nu
            )
            activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            friction_cone = crocoddyl.CostModelResidual(
                self.state, activation, residual
            )
            friction_name = self.pd.model.frames[i].name + "_friction_cone"
            costs.addCost(friction_name, friction_cone, self.pd.friction_cone_w)
            costs.changeCostStatus(friction_name, i in support_feet)

            name = self.pd.model.frames[i].name + "_foot_tracking"
            residual = crocoddyl.ResidualModelFrameTranslation(
                self.state, i, np.zeros(3), nu
            )
            foot_tracking = crocoddyl.CostModelResidual(self.state, residual)
            costs.addCost(name, foot_tracking, self.pd.foot_tracking_w)

            costs.changeCostStatus(name, False)

        control_residual = crocoddyl.ResidualModelControl(self.state, self.pd.uref)
        control_reg = crocoddyl.CostModelResidual(self.state, control_residual)
        costs.addCost("control_reg", control_reg, self.pd.control_reg_w)

        control_bound_residual = crocoddyl.ResidualModelControl(self.state, nu)
        control_bound_activation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(-self.pd.effort_limit, self.pd.effort_limit)
        )
        control_bound = crocoddyl.CostModelResidual(
            self.state, control_bound_activation, control_bound_residual
        )
        costs.addCost("control_bound", control_bound, self.pd.control_bound_w)

        self.update_tracking_costs(costs, tasks)
    
    def update_tracking_costs(self, costs, tasks):
        for i in self.pd.allContactIds:
            name = self.pd.model.frames[i].name + "_foot_tracking"
            index = 0
            if i in tasks[0]:
                costs.changeCostStatus(name, True)
                costs.costs[name].cost.residual.reference = tasks[1][index]
                index += 1
            else:
                costs.changeCostStatus(name, False)
