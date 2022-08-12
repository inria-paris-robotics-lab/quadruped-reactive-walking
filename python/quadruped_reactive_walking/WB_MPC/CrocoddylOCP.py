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

        self.initialize_models(gait, footsteps)

        self.x0 = self.pd.x0_reduced

        self.problem = crocoddyl.ShootingProblem(
            self.x0, self.models, self.terminal_model
        )
        self.ddp = crocoddyl.SolverFDDP(self.problem)

    def initialize_models(self, gait, footsteps):
        self.nodes = []
        for t, step in enumerate(gait):
            task = self.make_task(footsteps[t])
            support_feet = [self.pd.allContactIds[i] for i in np.nonzero(step == 1)[0]]
            self.nodes.append(Node(self.pd, self.state, support_feet, task))

        support_feet = [self.pd.allContactIds[i] for i in np.nonzero(gait[-1] == 1)[0]]
        self.terminal_node = Node(self.pd, self.state, support_feet, isTerminal=True)

        self.models = [node.model for node in self.nodes]
        self.terminal_model = self.terminal_node.model

    # def initialize_models(self, gait, footsteps):
    #     self.models = []
    #     for t, step in enumerate(gait):
    #         tasks = self.make_task(footsteps[t])
    #         support_feet = [self.pd.allContactIds[i] for i in np.nonzero(step == 1)[0]]
    #         self.models.append(self.create_model(support_feet, tasks))

    #     support_feet = [self.pd.allContactIds[i] for i in np.nonzero(gait[-1] == 1)[0]]
    #     self.terminal_model = self.create_model(support_feet, is_terminal=True)

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
        task = []
        for foot in range(4):
            if footstep[:, foot].any():
                task += [[self.pd.allContactIds[foot], footstep[:, foot]]]
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
        if tasks is not None:
            for (id, pose) in tasks:
                name = self.pd.model.frames[id].name + "_foot_tracking"
                if name in model.differential.costs.active.tolist():
                    model.differential.costs.costs[name].cost.residual.reference = pose
                else:
                    residual = crocoddyl.ResidualModelFrameTranslation(
                        self.state, id, pose, nu
                    )
                    cost = crocoddyl.CostModelResidual(self.state, residual)
                    model.differential.costs.addCost(
                        name, cost, self.pd.foot_tracking_w
                    )

        for i in self.pd.allContactIds:
            name = self.pd.model.frames[i].name + "_contact"
            model.differential.contacts.changeContactStatus(name, i in support_feet)

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
            name = (self.pd.model.frames[i].name + "_contact",)
            contact = crocoddyl.ContactModel3D(
                self.state, i, np.zeros(3), nu, np.zeros(2)
            )
            contacts.addContact(name, contact)
            if i not in support_feet:
                self.contactModel.changeContactStatus(name, False)

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
        self.costModel.addCost("terminal_velocity", state_cost, 1)

    def make_running_model(self, model, support_feet, tasks):
        """
        Add all the costs to the running models
        """
        nu = model.differential.actuation.nu
        costs = model.differential.costs
        for i in support_feet:
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
            friction_name = self.pd.model.frames[id].name + "_friction_cone"
            costs.addCost(friction_name, friction_cone, self.pd.friction_cone_w)

        control_residual = crocoddyl.ResidualModelControl(self.state, self.pd.uref)
        control_reg = crocoddyl.CostModelResidual(self.state, control_residual)
        self.costModel.addCost("control_reg", control_reg, self.pd.control_reg_w)

        control_bound_residual = crocoddyl.ResidualModelControl(self.state, nu)
        control_bound_activation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(-self.pd.effort_limit, self.pd.effort_limit)
        )
        control_bound = crocoddyl.CostModelResidual(
            self.state, control_bound_activation, control_bound_residual
        )
        costs.addCost("control_bound", control_bound, self.pd.control_bound_w)

        if tasks is not None:
            for (id, pose) in tasks:
                name = self.pd.model.frames[id].name + "_foot_tracking"
                residual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, id, pose, nu
                )
                foot_tracking = crocoddyl.CostModelResidual(self.state, residual)
                costs.addCost(name, foot_tracking, self.pd.foot_tracking_w)


class Node:
    def __init__(
        self, pd, state, supportFootIds=[], swingFootTask=[], isTerminal=False
    ):
        self.pd = pd
        self.isTerminal = isTerminal

        self.state = state
        if pd.useFixedBase == 0:
            self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        else:
            self.actuation = crocoddyl.ActuationModelFull(self.state)
        self.control = crocoddyl.ControlParametrizationModelPolyZero(self.actuation.nu)
        self.nu = self.actuation.nu

        self.createStandardModel(supportFootIds)
        if isTerminal:
            self.make_terminal_model()
        else:
            self.make_running_model(supportFootIds, swingFootTask)

    def createStandardModel(self, supportFootIds):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        self.contactModel = crocoddyl.ContactModelMultiple(self.state, self.nu)
        for i in self.pd.allContactIds:
            supportContactModel = crocoddyl.ContactModel3D(
                self.state, i, np.array([0.0, 0.0, 0.0]), self.nu, np.array([0.0, 0.0])
            )

            self.contactModel.addContact(
                self.pd.model.frames[i].name + "_contact", supportContactModel
            )

            if i not in supportFootIds:
                self.contactModel.changeContactStatus(
                    self.pd.model.frames[i].name + "_contact", False
                )

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.nu)

        stateResidual = crocoddyl.ResidualModelState(self.state, self.pd.xref, self.nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            self.pd.state_reg_w**2
        )
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1)

        self.costModel = costModel

        self.dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, self.contactModel, self.costModel, 0.0, True
        )
        self.model = crocoddyl.IntegratedActionModelEuler(
            self.dmodel, self.control, self.pd.dt
        )

    def make_terminal_model(self):
        self.isTerminal = True
        stateResidual = crocoddyl.ResidualModelState(self.state, self.pd.xref, self.nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(
            self.pd.terminal_velocity_w**2
        )
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        self.costModel.addCost("terminalVelocity", stateReg, 1)

    def make_running_model(self, supportFootIds, swingFootTask):
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.pd.Rsurf, self.pd.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, i, cone, self.nu
            )
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            frictionCone = crocoddyl.CostModelResidual(
                self.state, coneActivation, coneResidual
            )
            self.costModel.addCost(
                self.pd.model.frames[i].name + "_frictionCone",
                frictionCone,
                self.pd.friction_cone_w,
            )

        ctrlResidual = crocoddyl.ResidualModelControl(self.state, self.pd.uref)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        self.costModel.addCost("ctrlReg", ctrlReg, self.pd.control_reg_w)

        ctrl_bound_residual = crocoddyl.ResidualModelControl(self.state, self.nu)
        ctrl_bound_activation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(-self.pd.effort_limit, self.pd.effort_limit)
        )
        ctrl_bound = crocoddyl.CostModelResidual(
            self.state, ctrl_bound_activation, ctrl_bound_residual
        )
        self.costModel.addCost("ctrlBound", ctrl_bound, self.pd.control_bound_w)

        self.tracking_cost(swingFootTask)

    def remove_running_costs(self):
        runningCosts = self.dmodel.costs.active.tolist()
        idx = runningCosts.index("stateReg")
        runningCosts.pop(idx)
        for cost in runningCosts:
            if cost in self.dmodel.costs.active.tolist():
                self.dmodel.costs.removeCost(cost)

    def remove_terminal_cost(self):
        if "terminalVelocity" in self.dmodel.costs.active.tolist():
            self.dmodel.costs.removeCost("terminalVelocity")

    def tracking_cost(self, task):
        if task is not None:
            for (id, pose) in task:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, id, pose, self.nu
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, frameTranslationResidual
                )
                if (
                    self.pd.model.frames[id].name + "_foot_tracking"
                    in self.dmodel.costs.active.tolist()
                ):
                    self.dmodel.costs.removeCost(
                        self.pd.model.frames[id].name + "_foot_tracking"
                    )
                self.costModel.addCost(
                    self.pd.model.frames[id].name + "_foot_tracking",
                    footTrack,
                    self.pd.foot_tracking_w,
                )
