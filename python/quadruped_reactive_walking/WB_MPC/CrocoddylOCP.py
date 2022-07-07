from tracemalloc import start
from .ProblemData import ProblemData
from .Target import Target
import crocoddyl
import pinocchio as pin
import numpy as np
from time import time

class OCP:
    def __init__(self, pd:ProblemData, target:Target):
        self.pd = pd
        self.target = target
        self.state = crocoddyl.StateMultibody(self.pd.model)
        if pd.useFixedBase == 0:
            self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        else:
            self.actuation = crocoddyl.ActuationModelFull(self.state)

    def make_crocoddyl_ocp(self, x0):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        """
        self.control = crocoddyl.ControlParametrizationModelPolyZero(
            self.actuation.nu)

        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pin.forwardKinematics(self.pd.model, self.pd.rdata, q0)
        pin.updateFramePlacements(self.pd.model, self.pd.rdata)

        model = []
        for t in range(self.pd.T):
            target = self.target.evaluate_in_t(t)
            freeIds = [idf for idf in self.pd.allContactIds if idf not in self.target.contactSequence[t]]
            contactIds = self.target.contactSequence[t]
            model += self.createFootstepModels(target, contactIds, freeIds)

        freeIds = [idf for idf in self.pd.allContactIds if idf not in self.target.contactSequence[self.pd.T]]
        contactIds = self.target.contactSequence[self.pd.T]
        model += self.createFootstepModels(self.target.evaluate_in_t(self.pd.T), contactIds, freeIds, True)

        problem = crocoddyl.ShootingProblem(x0, model[:-1], model[-1])

        return problem

    def createFootstepModels(self,  target, supportFootIds,
                             swingFootIds, isTerminal=False):
        """ Action models for a footstep phase.
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        # Action models for the foot swing
        footSwingModel = []
        swingFootTask = []
        for i in swingFootIds:
            try:
                tref = target[i]
                swingFootTask += [[i, pin.SE3(np.eye(3), tref)]]
            except:
                pass

        footSwingModel += [self.createSwingFootModel(supportFootIds, swingFootTask=swingFootTask, isTerminal=isTerminal)]
        return footSwingModel

    def createSwingFootModel(self, supportFootIds, swingFootTask=None, isTerminal=False):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(self.state, i, np.array([0., 0., 0.]), nu,
                                                           np.array([0., 0.]))
            contactModel.addContact(self.pd.model.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)

        if not isTerminal:

            for i in supportFootIds:
                cone = crocoddyl.FrictionCone(self.pd.Rsurf, self.pd.mu, 4, False)
                coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
                coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
                frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
                costModel.addCost(self.pd.model.frames[i].name + "_frictionCone", frictionCone, self.pd.friction_cone_w)
            if swingFootTask is not None:
                for i in swingFootTask:
                    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,nu)
                    footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                    costModel.addCost(self.pd.model.frames[i[0]].name + "_footTrack", footTrack, self.pd.foot_tracking_w)

            ctrlResidual = crocoddyl.ResidualModelControl(self.state, self.pd.uref)
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
            costModel.addCost("ctrlReg", ctrlReg, self.pd.control_reg_w)

            ctrl_bound_residual = crocoddyl.ResidualModelControl(self.state, nu)
            ctrl_bound_activation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-self.pd.effort_limit, self.pd.effort_limit))
            ctrl_bound = crocoddyl.CostModelResidual(self.state, ctrl_bound_activation, ctrl_bound_residual)
            costModel.addCost("ctrlBound", ctrl_bound, self.pd.control_bound_w)

        stateResidual = crocoddyl.ResidualModelState(self.state, self.pd.xref, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(self.pd.state_reg_w**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        costModel.addCost("stateReg", stateReg, 1)

        if isTerminal:
            stateResidual = crocoddyl.ResidualModelState(self.state, self.pd.xref, nu)
            stateActivation = crocoddyl.ActivationModelWeightedQuad(self.pd.terminal_velocity_w**2)
            stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
            costModel.addCost("terminalVelocity", stateReg, 1)

        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)

        model = crocoddyl.IntegratedActionModelEuler(
            dmodel, self.control, self.pd.dt)

        return model

# Solve
    def solve(self, x0, guess=None):
        problem = self.make_crocoddyl_ocp(x0)
        self.ddp = crocoddyl.SolverFDDP(problem)
        self.ddp.setCallbacks([crocoddyl.CallbackVerbose()])

        if not guess:
            print("No warmstart provided")
            xs = [x0] * (self.ddp.problem.T + 1)
            us = self.ddp.problem.quasiStatic([x0] * self.ddp.problem.T)
        else:
            xs = guess['xs']
            us = guess['us']
            print("Using warmstart")
        start_time = time()
        self.ddp.solve(xs, us, 1, False)
        print("Solver time: ", time()- start_time, "\n")

    def get_croco_forces(self):
        d = self.ddp.problem.runningDatas[0]
        cnames = d.differential.multibody.contacts.contacts.todict().keys()
        forces = {n: [] for n in cnames}

        for m in self.ddp.problem.runningDatas:
            mdict = m.differential.multibody.contacts.contacts.todict()
            for n in cnames:
                if n in mdict:
                    forces[n] += [(mdict[n].jMf.inverse()*mdict[n].f).linear]
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
                f_tmp += [(mdict[n].jMf.inverse()*mdict[n].f).linear]
            forces += [np.concatenate(f_tmp)]
        return forces

    def get_croco_acc(self):
        acc = []
        [acc.append(m.differential.xout)
         for m in self.ddp.problem.runningDatas]
        return acc

    def get_results(self):
        x = self.ddp.xs.tolist()
        a = self.get_croco_acc()
        u = self.ddp.us.tolist()
        if self.pd.useFixedBase == 0:
            f_ws = self.get_croco_forces_ws()
        else:
            f_ws = []

        return None, x, a, u, f_ws, None
