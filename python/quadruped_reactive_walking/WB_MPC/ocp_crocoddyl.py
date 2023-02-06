from .problem_data import ProblemData
from .Target import Target
import crocoddyl
import sobec
import pinocchio as pin
import numpy as np
from time import time
from .ocp_abstract import OCPAbstract


class CrocOCP(OCPAbstract):
    def __init__(self, pd: ProblemData, params, footsteps, base_refs, **kwargs):
        super().__init__(pd, params)

        self.state = crocoddyl.StateMultibody(self.pd.model)

        # Set the problem parameters
        self.initialized = False
        self.t_problem_update = 0
        self.t_update_last_model = 0.0
        self.t_shift = 0.0

        params = self.params
        self.life_gait = params.gait
        self.starting_gait = np.array([[1, 1, 1, 1]] * params.starting_nodes)
        self.ending_gait = np.array([[1, 1, 1, 1]] * params.ending_nodes)
        self.initialization_gait = np.concatenate(
            [self.starting_gait, self.life_gait, self.ending_gait]
        )
        self.current_gait = np.append(
            self.starting_gait, self.ending_gait[0].reshape(1, -1), axis=0
        )
        self.x0 = self.pd.x0

        self.life_rm, self.life_tm = self.initialize_models(
            self.life_gait, footsteps, base_refs
        )
        self.start_rm, self.start_tm = self.initialize_models(self.ending_gait)
        self.end_rm, self.end_tm = self.initialize_models(self.ending_gait)

        self.problem = crocoddyl.ShootingProblem(self.x0, self.start_rm, self.start_tm)
        self.ddp = crocoddyl.SolverFDDP(self.problem)
        if params.ocp.verbose:
            self.ddp.setCallbacks([crocoddyl.CallbackVerbose()])

    def initialize_models(self, gait, footsteps=[], base_refs=[]):
        models = []
        for t, step in enumerate(gait):
            tasks = self.make_task(footsteps[t]) if footsteps else []
            base_task = base_refs[t] if base_refs else []
            support_feet = [self.pd.feet_ids[i] for i in np.nonzero(step == 1)[0]]
            switch_matrix = (
                gait[t] if ((gait[t] != gait[t - 1]).any() or t == 0) else []
            )
            switch_feet = [
                self.pd.feet_ids[i] for i in np.nonzero(switch_matrix == 1)[0]
            ]
            models.append(
                self.create_model(support_feet, switch_feet, tasks, base_task)
            )

        support_feet = [self.pd.feet_ids[i] for i in np.nonzero(gait[-1] == 1)[0]]
        terminal_model = self.create_model(support_feet, is_terminal=True)

        return models, terminal_model

    def solve(self, k, x0, footstep, base_ref, xs_init=None, us_init=None):
        t_start = time()
        self.x0 = x0
        self.make_ocp(k, footstep, base_ref)

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

        self.ddp.solve(xs, us, self.max_iter, False)

        t_ddp = time()
        self.t_ddp = t_ddp - t_warm_start

        self.t_solve = time() - t_start
        self.num_iters = self.ddp.iter

    def make_ocp(self, k, footstep, base_task):
        """
        Create a shooting problem for a simple walking gait.

        :param x0: initial state
        """
        pin.forwardKinematics(self.pd.model, self.pd.rdata, self.x0[: self.pd.nq])
        pin.updateFramePlacements(self.pd.model, self.pd.rdata)

        if self.initialized:
            tasks = self.make_task(footstep)
            t = int(k / self.params.mpc_wbc_ratio) - 1

            if t < len(self.start_rm):
                support_feet = [
                    self.pd.feet_ids[i] for i in np.nonzero(self.life_gait[t] == 1)[0]
                ]
                m = self.life_rm[t]
                self.current_gait = np.insert(
                    self.current_gait[1:], -1, self.life_gait[t].reshape(1, -1), axis=0
                )

            elif (
                t
                < len(self.start_rm) + len(self.life_rm) * self.params.gait_repetitions
            ):
                self.life_gait = np.roll(self.life_gait, -1, axis=0)
                support_feet = [
                    self.pd.feet_ids[i] for i in np.nonzero(self.life_gait[-1] == 1)[0]
                ]
                m = self.problem.runningModels[0]
                self.current_gait = np.insert(
                    self.current_gait[1:], -1, self.life_gait[-1].reshape(1, -1), axis=0
                )

            else:
                i = (
                    0
                    if t
                    == len(self.start_rm)
                    + len(self.life_rm) * self.params.gait_repetitions
                    else 1
                )
                # choose to pich the node with impact or not
                support_feet = [
                    self.pd.feet_ids[i] for i in np.nonzero(self.ending_gait[i] == 1)[0]
                ]

                if i:
                    m = self.end_rm[1]
                    self.current_gait = np.insert(
                        self.current_gait[1:],
                        -1,
                        self.ending_gait[1].reshape(1, -1),
                        axis=0,
                    )
                else:
                    m = self.end_rm[0]
                    self.current_gait = np.insert(
                        self.current_gait[1:],
                        -1,
                        self.ending_gait[0].reshape(1, -1),
                        axis=0,
                    )
                base_task = []

            self.update_model(m, tasks, base_task, support_feet)
            self.circular_append(m)

        self.problem.x0 = self.x0

        self.initialized = True

    def circular_append(self, m):
        d = m.createData()
        self.problem.circularAppend(m, d)

    def get_results(self):
        return (
            self.current_gait.copy(),
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

    def update_model(self, model, tasks, base_task, support_feet, is_terminal=False):
        for i in self.pd.feet_ids:
            name = self.pd.model.frames[i].name + "_contact"
            model.differential.contacts.changeContactStatus(name, i in support_feet)
        if not is_terminal:
            self.update_tracking_costs(
                model.differential.costs, tasks, base_task, support_feet
            )

    def create_model(
        self, support_feet=[], switch_feet=[], tasks=[], base_task=[], is_terminal=False
    ):
        """
        Create the action model

        :param state: swinging foot task
        :param support_feet: list of support feet ids
        :param task: list of support feet ids and associated tracking reference
        :param isTterminal: true for the terminal node
        :return action model for a swing foot phase
        """
        pin.forwardKinematics(self.pd.model, self.pd.rdata, self.pd.q0)
        pin.updateFramePlacements(self.pd.model, self.pd.rdata)

        model = self.create_standard_model(support_feet)
        if is_terminal:
            self.make_terminal_model(model)
        else:
            self.make_running_model(model, support_feet, switch_feet, tasks, base_task)

        return model

    def create_standard_model(self, support_feet):
        """
        Create a standard action model

        :param state: swinging foot task
        :param support_feet: list of support feet ids
        :return action model for a swing foot phase
        """
        actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        nu = actuation.nu

        control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        contacts = sobec.ContactModelMultiple(self.state, nu)
        for i in self.pd.feet_ids:
            name = self.pd.model.frames[i].name + "_contact"
            contact = sobec.ContactModel3D(
                self.state,
                i,
                np.zeros(3),
                nu,
                self.pd.baumgarte_gains,
                pin.LOCAL_WORLD_ALIGNED,
            )
            contacts.addContact(name, contact)
            if i not in support_feet:
                contacts.changeContactStatus(name, False)

        costs = crocoddyl.CostModelSum(self.state, nu)
        residual = crocoddyl.ResidualModelState(self.state, self.pd.xref, nu)
        activation = crocoddyl.ActivationModelWeightedQuad(self.pd.state_reg_w**2)
        state_cost = crocoddyl.CostModelResidual(self.state, activation, residual)
        costs.addCost("state_reg", state_cost, 1)

        state_bound_residual = crocoddyl.ResidualModelState(
            self.state, self.pd.xref, nu
        )
        activation = crocoddyl.ActivationModelWeightedQuadraticBarrier(
            crocoddyl.ActivationBounds(-self.pd.state_limit, self.pd.state_limit),
            self.pd.state_bound_w**2,
        )
        state_bound_cost = crocoddyl.CostModelResidual(
            self.state, activation, state_bound_residual
        )
        costs.addCost("state_limitBound", state_bound_cost, 1)

        differential = sobec.DifferentialActionModelContactFwdDynamics(
            self.state, actuation, contacts, costs, 0.0, True
        )
        model = crocoddyl.IntegratedActionModelEuler(
            differential, control, self.params.dt_mpc
        )

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

    def make_running_model(self, model, support_feet, switch_feet, tasks, base_task):
        """
        Add all the costs to the running models
        """
        nu = model.differential.actuation.nu
        costs = model.differential.costs
        for i in self.pd.feet_ids:
            start_pos = self.pd.rdata.oMf[i].translation

            # Contact forces
            cone = crocoddyl.FrictionCone(self.pd.Rsurf, self.pd.mu, 4, False, 3)
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

            name = "%s_forceReg" % self.pd.model.frames[i].name
            nc = len(model.differential.contacts.active_set)
            ref_force = np.array([0, 0, self.pd.robot_weight / nc])
            ref_Force = pin.Force(ref_force, ref_force * 0)
            forceRegResidual = sobec.ResidualModelContactForce(
                self.state, i, ref_Force, 3, nu
            )
            forceRegCost = crocoddyl.CostModelResidual(self.state, forceRegResidual)
            costs.addCost(name, forceRegCost, self.pd.force_reg_w)
            costs.changeCostStatus(name, False)

            # Tracking foot trajectory
            name = self.pd.model.frames[i].name + "_foot_tracking"
            residual = crocoddyl.ResidualModelFrameTranslation(
                self.state, i, np.zeros(3), nu
            )
            foot_tracking = crocoddyl.CostModelResidual(self.state, residual)
            if self.pd.foot_tracking_w > 0:
                costs.addCost(name, foot_tracking, self.pd.foot_tracking_w)
                costs.changeCostStatus(name, False)

            # Swing foot
            groundColRes = crocoddyl.ResidualModelFrameTranslation(
                self.state, i, start_pos, nu
            )

            groundColBounds = crocoddyl.ActivationBounds(
                np.array([-1000, -1000, start_pos[2]]), np.array([1000, 1000, 1000])
            )
            groundColAct = crocoddyl.ActivationModelQuadraticBarrier(groundColBounds)
            groundColCost = crocoddyl.CostModelResidual(
                self.state, groundColAct, groundColRes
            )

            name = "%s_groundCol" % self.pd.model.frames[i].name
            if self.pd.ground_collision_w > 0:
                costs.addCost(
                    name,
                    groundColCost,
                    self.pd.ground_collision_w,
                )
            costs.changeCostStatus(name, False)

            flyHighResidual = sobec.ResidualModelFlyHigh(
                self.state, i, self.pd.fly_high_slope / 2.0, nu
            )
            flyHighCost = crocoddyl.CostModelResidual(self.state, flyHighResidual)

            name = "%s_flyHigh" % self.pd.model.frames[i].name
            if self.pd.fly_high_w > 0:
                costs.addCost(
                    name,
                    flyHighCost,
                    self.pd.fly_high_w,
                )
            costs.changeCostStatus(name, False)

            vertical_velocity_reg_residual = crocoddyl.ResidualModelFrameVelocity(
                self.state,
                i,
                pin.Motion.Zero(),
                pin.ReferenceFrame.WORLD,
                nu,
            )
            vertical_velocity_activation = crocoddyl.ActivationModelWeightedQuad(
                np.array([0, 0, 1, 0, 0, 0])
            )

            name = "%s_vel_zReg" % self.pd.model.frames[i].name
            vertical_velocity_reg_cost = crocoddyl.CostModelResidual(
                self.state,
                vertical_velocity_activation,
                vertical_velocity_reg_residual,
            )
            costs.addCost(
                name,
                vertical_velocity_reg_cost,
                self.pd.vertical_velocity_reg_w,
            )

            # Fake impoact
            if i in switch_feet and i in support_feet:
                impactResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, i, start_pos, nu
                )
                impactAct = crocoddyl.ActivationModelWeightedQuad(np.array([0, 0, 1]))
                impactCost = crocoddyl.CostModelResidual(
                    self.state, impactAct, impactResidual
                )
                if self.pd.impact_altitude_w > 0:
                    costs.addCost(
                        "%s_altitudeImpact" % self.pd.model.frames[i].name,
                        impactCost,
                        self.pd.impact_altitude_w / self.params.dt_mpc,
                    )

                impactVelResidual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    i,
                    pin.Motion.Zero(),
                    pin.ReferenceFrame.WORLD,
                    nu,
                )
                if self.pd.impact_velocity_w > 0:
                    impactVelCost = crocoddyl.CostModelResidual(
                        self.state, impactVelResidual
                    )
                    costs.addCost(
                        "%s_velimpact" % self.pd.model.frames[i].name,
                        impactVelCost,
                        self.pd.impact_velocity_w / self.params.dt_mpc,
                    )

        name = "base_velocity_tracking"
        if list(base_task):
            ref = pin.Motion(base_task[:3], base_task[3:])
        else:
            ref = pin.Motion.Zero()

        residual_base_velocity = crocoddyl.ResidualModelFrameVelocity(
            self.state, self.pd.base_id, ref, pin.LOCAL, nu
        )
        base_velocity = crocoddyl.CostModelResidual(self.state, residual_base_velocity)

        if self.pd.base_velocity_tracking_w > 0:
            costs.addCost(name, base_velocity, self.pd.base_velocity_tracking_w)

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

        self.update_tracking_costs(costs, tasks, base_task, support_feet)

    def update_tracking_costs(self, costs, tasks, base_task, support_feet):
        index = 0
        for i in self.pd.feet_ids:
            if self.pd.foot_tracking_w > 0:
                name = self.pd.model.frames[i].name + "_foot_tracking"
                if i in tasks[0]:
                    costs.costs[name].cost.residual.reference = tasks[1][index]
                    index += 1
                costs.changeCostStatus(name, i not in support_feet)

            name = "%s_forceReg" % self.pd.model.frames[i].name
            costs.changeCostStatus(name, i in support_feet)

            name = "%s_groundCol" % self.pd.model.frames[i].name
            costs.changeCostStatus(name, i not in support_feet)

            name = "%s_flyHigh" % self.pd.model.frames[i].name
            costs.changeCostStatus(name, i not in support_feet)

            name = "%s_vel_zReg" % self.pd.model.frames[i].name
            costs.changeCostStatus(name, i not in support_feet)

        if list(base_task) and self.pd.base_velocity_tracking_w > 0:
            name = "base_velocity_tracking"
            ref = pin.Motion(base_task[:3], base_task[3:])
            costs.costs[name].cost.residual.reference = ref
