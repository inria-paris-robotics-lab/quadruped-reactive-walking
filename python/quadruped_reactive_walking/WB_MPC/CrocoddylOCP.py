from tracemalloc import start

from matplotlib.pyplot import switch_backend
from .ProblemData import ProblemData
from .Target import Target
from .OcpResult import OcpResult
import crocoddyl
import pinocchio as pin
import numpy as np
from time import time


class OCP:
    def __init__(self, pd: ProblemData, target: Target):
        self.pd = pd
        self.target = target

        self.results = OcpResult()
        self.state = crocoddyl.StateMultibody(self.pd.model)
        self.initialized = False
        self.initialize_models()

        self.x0 = self.pd.x0_reduced
        self.t_update_last_model = 0

    def initialize_models(self):
        self.nodes = []
        for t in range(self.pd.T):
            task = self.make_task(self.target.evaluate_in_t(t), self.target.contactSequence[t])
            self.nodes.append(
                Node(self.pd, self.state, self.target.contactSequence[t], task))
        self.terminal_node = Node(self.pd, self.state, self.target.contactSequence[self.pd.T], isTerminal=True)

        self.models = [node.model for node in self.nodes]
        self.terminal_model = self.terminal_node.model

    def make_ocp(self):
        """
        Create a shooting problem for a simple walking gait.

        :param x0: initial state
        """
        t_start = time()

        # Compute the current foot positions
        q0 = self.x0[: self.pd.nq]
        pin.forwardKinematics(self.pd.model, self.pd.rdata, q0)
        pin.updateFramePlacements(self.pd.model, self.pd.rdata)

        t_FK = time()
        self.t_FK = t_FK - t_start

        if self.initialized:
            task = self.make_task(self.target.evaluate_in_t(self.pd.T-1), self.target.contactSequence[self.pd.T-1]) # model without contact for this task
            self.nodes[0].update_model(self.target.contactSequence[self.pd.T-1], task)
            #self.problem.circularAppend(self.nodes[0].model, self.nodes[0].model.createData())
            self.nodes = self.nodes[1:] + [self.nodes[0]]
            self.models = [node.model for node in self.nodes]

        t_shift = time()
        self.t_shift = t_shift - t_FK

        t_update_last_model = time()
        self.t_update_last_model = t_update_last_model - t_shift

        # If you need update terminal model
        t_update_terminal_model = time()
        self.t_update_terminal_model = t_update_terminal_model - t_shift


        self.problem = crocoddyl.ShootingProblem(
            self.x0, self.models, self.terminal_model
        )
        self.ddp = crocoddyl.SolverFDDP(self.problem)


        self.initialized = True

    #def update_model(self, node, target, contactIds, isTerminal=False):
    #    """
    #    Action models for a footstep phase.
    #    :return footstep action models
    #    """
    #    # update translation
    #    translation = np.zeros(3)
#
    #    node.update_model(contactIds, translation, isTerminal=isTerminal)

    def solve(self, x0, guess=None):

        self.x0 = x0

        t_start = time()

        self.make_ocp()

        t_update = time()
        self.t_update = t_update - t_start

        if not guess:
            xs = [x0] * (self.ddp.problem.T + 1)
            us = self.ddp.problem.quasiStatic([x0] * self.ddp.problem.T)
        else:
            xs = guess["xs"]
            us = guess["us"]

        t_warm_start = time()
        self.t_warm_start = t_warm_start - t_update

        self.ddp.setCallbacks([crocoddyl.CallbackVerbose()])
        self.ddp.solve(xs, us, 1, False)

        t_ddp = time()
        self.t_ddp = t_ddp - t_warm_start

        self.t_solve = time() - t_start

    def make_task(self, target, contactIds):
        swingFootTask = []
        for i in self.freeIds_from_contactIds(contactIds):
            try:
                tref = target[i]
                swingFootTask += [[i, pin.SE3(np.eye(3), tref)]]
            except:
                pass
        return swingFootTask

    def freeIds_from_contactIds(self, contactIds):
        return [idf for idf in self.pd.allContactIds if idf not in contactIds]

    def get_results(self):
        self.results.x = self.ddp.xs.tolist()
        self.results.a = self.get_croco_acc()
        self.results.u = self.ddp.us.tolist()
        self.results.K = self.ddp.K
        self.results.solver_time = self.t_solve
        return self.results

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
        [acc.append(m.differential.xout)
         for m in self.ddp.problem.runningDatas]
        return acc


class Node:
    def __init__(self, pd, state, supportFootIds=[], swingFootTask = [], isTerminal=False):
        self.pd = pd
        self.isTerminal = isTerminal

        self.state = state
        if pd.useFixedBase == 0:
            self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        else:
            self.actuation = crocoddyl.ActuationModelFull(self.state)
        self.control = crocoddyl.ControlParametrizationModelPolyZero(
            self.actuation.nu)
        self.nu = self.actuation.nu

        self.createStandardModel(supportFootIds)
        if isTerminal:
            self.make_terminal_model()
        else:
            self.make_running_model(supportFootIds ,swingFootTask)

    def createStandardModel(self, supportFootIds):
        """Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """

        self.contactModel = crocoddyl.ContactModelMultiple(self.state, self.nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(
                self.state, i, np.array(
                    [0.0, 0.0, 0.0]), self.nu, np.array([0.0, 0.0])
            )
            self.contactModel.addContact(
                self.pd.model.frames[i].name + "_contact", supportContactModel
            )

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.nu)

        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.pd.xref, self.nu)
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

    def update_contact_model(self, supportFootIds):
        self.remove_contacts()
        self.contactModel = crocoddyl.ContactModelMultiple(self.state, self.nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(
                self.state, i, np.array(
                    [0.0, 0.0, 0.0]), self.nu, np.array([0.0, 0.0])
            )
            self.dmodel.contacts.addContact(
                self.pd.model.frames[i].name + "_contact", supportContactModel
            )

    def make_terminal_model(self):
        self.isTerminal = True
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.pd.xref, self.nu)
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

        ctrl_bound_residual = crocoddyl.ResidualModelControl(
            self.state, self.nu)
        ctrl_bound_activation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(-self.pd.effort_limit,
                                       self.pd.effort_limit)
        )
        ctrl_bound = crocoddyl.CostModelResidual(
            self.state, ctrl_bound_activation, ctrl_bound_residual
        )
        self.costModel.addCost("ctrlBound", ctrl_bound,
                               self.pd.control_bound_w)


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

    def remove_contacts(self):
        allContacts = self.dmodel.contacts.contacts.todict()
        for c in allContacts:
            self.dmodel.contacts.removeContact(c)

    def tracking_cost(self, swingFootTask):

        #  TODO all this done in initialisation ? (we always have at maximum one swing
        # foot task so you can just initialize i once and then add or remove it)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, i[0], i[1].translation, self.nu
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, frameTranslationResidual
                )
                if (
                    self.pd.model.frames[i[0]].name + "_footTrack"
                    in self.dmodel.costs.active.tolist()
                ):
                    self.dmodel.costs.removeCost(
                        self.pd.model.frames[i[0]].name + "_footTrack"
                    )
                self.costModel.addCost(
                    self.pd.model.frames[i[0]].name + "_footTrack",
                    footTrack,
                    self.pd.foot_tracking_w,
                )

    def update_model(self, supportFootIds=[], swingFootTask=[]):
        if self.isTerminal:
            self.update_contact_model(supportFootIds)
        else:
            self.update_contact_model(supportFootIds)
            self.tracking_cost(swingFootTask)
