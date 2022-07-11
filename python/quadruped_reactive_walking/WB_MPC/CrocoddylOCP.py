from tracemalloc import start
from .ProblemData import ProblemData
from .Target import Target
from .OcpResult import OcpResult
import crocoddyl
import pinocchio as pin
import numpy as np
from time import time

class OCP:
    def __init__(self, pd:ProblemData, target:Target):
        self.pd = pd
        self.target = target
        
        self.results = OcpResult()
        self.state = crocoddyl.StateMultibody(self.pd.model)
        self.initialize_models()
    
    def initialize_models(self):
        self.models = []
        for _ in range(self.pd.T):
            self.models.append(Model(self.pd, self.state)) # RunningModels
        self.models.append(Model(self.pd, self.state, isTerminal=True)) #TerminalModel
        

    def make_crocoddyl_ocp(self, x0):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        """
        

        # Compute the current foot positions
        q0 = x0[:self.pd.nq]
        pin.forwardKinematics(self.pd.model, self.pd.rdata, q0)
        pin.updateFramePlacements(self.pd.model, self.pd.rdata)

        for t in range(self.pd.T):
            target = self.target.evaluate_in_t(t)
            freeIds = [idf for idf in self.pd.allContactIds if idf not in self.target.contactSequence[t]]
            contactIds = self.target.contactSequence[t]
            self.appendTargetToModel(self.models[t], target, freeIds, False)

        freeIds = [idf for idf in self.pd.allContactIds if idf not in self.target.contactSequence[self.pd.T]]
        #contactIds = self.target.contactSequence[self.pd.T]
        self.appendTargetToModel(self.models[self.pd.T], self.target.evaluate_in_t(self.pd.T), freeIds, True)

        problem = crocoddyl.ShootingProblem(x0, 
                                            [m.model for m in self.models[:-1]], 
                                            self.models[-1].model)

        return problem

    def appendTargetToModel(self, model, target, swingFootIds, isTerminal=False):
        """ Action models for a footstep phase.
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        # Action models for the foot swing
        swingFootTask = []
        for i in swingFootIds:
            try:
                tref = target[i]
                swingFootTask += [[i, pin.SE3(np.eye(3), tref)]]
            except:
                pass
        
        model.tracking_cost(swingFootTask)


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

    def get_results(self):
        self.results.x = self.ddp.xs.tolist()
        self.results.a = self.get_croco_acc()
        self.results.u = self.ddp.us.tolist()
        self.results.K = self.ddp.K
        return self.results

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

    

class Model:
    def __init__(self, pd, state, supportFootIds=[], isTerminal=False):
        self.pd = pd
        self.supportFootIds=supportFootIds
        self.isTerminal=isTerminal

        self.state = state
        if pd.useFixedBase == 0:
            self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        else:
            self.actuation = crocoddyl.ActuationModelFull(self.state)
        self.control = crocoddyl.ControlParametrizationModelPolyZero(self.actuation.nu)
        self.nu = self.actuation.nu

        self.createStandardModel()

    def createStandardModel(self):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot

        
        self.contactModel = crocoddyl.ContactModelMultiple(self.state, self.nu)
        for i in self.supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(self.state, i, np.array([0., 0., 0.]), self.nu,
                                                           np.array([0., 0.]))
            self.contactModel.addContact(self.pd.model.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, self.nu)

        if not self.isTerminal:

            for i in self.supportFootIds:
                cone = crocoddyl.FrictionCone(self.pd.Rsurf, self.pd.mu, 4, False)
                coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, self.nu)
                coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
                frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
                costModel.addCost(self.pd.model.frames[i].name + "_frictionCone", frictionCone, self.pd.friction_cone_w)

            ctrlResidual = crocoddyl.ResidualModelControl(self.state, self.pd.uref)
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
            costModel.addCost("ctrlReg", ctrlReg, self.pd.control_reg_w)

            ctrl_bound_residual = crocoddyl.ResidualModelControl(self.state, self.nu)
            ctrl_bound_activation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-self.pd.effort_limit, self.pd.effort_limit))
            ctrl_bound = crocoddyl.CostModelResidual(self.state, ctrl_bound_activation, ctrl_bound_residual)
            costModel.addCost("ctrlBound", ctrl_bound, self.pd.control_bound_w)

        stateResidual = crocoddyl.ResidualModelState(self.state, self.pd.xref, self.nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(self.pd.state_reg_w**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        costModel.addCost("stateReg", stateReg, 1)

        if self.isTerminal:
            stateResidual = crocoddyl.ResidualModelState(self.state, self.pd.xref, self.nu)
            stateActivation = crocoddyl.ActivationModelWeightedQuad(self.pd.terminal_velocity_w**2)
            stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
            costModel.addCost("terminalVelocity", stateReg, 1)

        self.costModel = costModel

        self.dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, self.contactModel,
                                                                    self.costModel, 0., True)
        self.model = crocoddyl.IntegratedActionModelEuler(self.dmodel, self.control, self.pd.dt)

    def tracking_cost(self, swingFootTask):
        if swingFootTask is not None and not self.isTerminal:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,self.nu)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                if self.pd.model.frames[i[0]].name + "_footTrack" in self.dmodel.costs.active.tolist():
                    self.dmodel.costs.removeCost(self.pd.model.frames[i[0]].name + "_footTrack")
                self.costModel.addCost(self.pd.model.frames[i[0]].name + "_footTrack", footTrack, self.pd.foot_tracking_w)



