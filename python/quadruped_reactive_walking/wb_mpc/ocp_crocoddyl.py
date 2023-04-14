import crocoddyl
import sobec
import pinocchio as pin
import numpy as np
from time import time
from .ocp_abstract import OCPAbstract
from typing import Optional
from ..tools.utils import no_copy_roll, no_copy_roll_insert


class CrocOCP(OCPAbstract):
    """
    Generate a Crocoddyl OCP for the control task.
    """

    def __init__(self, params, footsteps, base_refs):
        super().__init__(params)

        self.state = crocoddyl.StateMultibody(self.task.model)

        self.rdata = self.task.model.createData()

        # Set the problem parameters
        self.t_problem_update = 0
        self.t_update_last_model = 0.0
        self.t_shift = 0.0

        params = self.params
        self.life_gait = params.gait
        self.starting_gait = np.ones((params.starting_nodes, 4), dtype=np.int32)
        self.ending_gait = np.ones((params.ending_nodes, 4), dtype=np.int32)
        self.current_gait = np.append(
            self.starting_gait, self.ending_gait[0].reshape(1, -1), axis=0
        )
        self.x0 = self.task.x0

        self.life_rm, self.life_tm = self.initialize_models_from_gait(
            self.life_gait, footsteps, base_refs
        )
        self.start_rm, self.start_tm = self.initialize_models_from_gait(
            self.starting_gait
        )
        self.end_rm, self.end_tm = self.initialize_models_from_gait(self.ending_gait)

        self.problem = crocoddyl.ShootingProblem(self.x0, self.start_rm, self.start_tm)
        self.ddp = crocoddyl.SolverFDDP(self.problem)
        if params.ocp.verbose:
            self.ddp.setCallbacks([crocoddyl.CallbackVerbose()])

    def get_type_str():
        return "croc"

    def initialize_models_from_gait(self, gait, footsteps=None, base_refs=None):
        """Create action models (problem stages) from a gait matrix and other optional data."""
        assert (footsteps is None) == (
            base_refs is None
        )  # both or neither must be none
        if footsteps is not None:
            assert len(footsteps) == len(base_refs)
        running_models = []
        feet_ids = np.asarray(self.task.feet_ids)
        for t in range(gait.shape[0]):
            support_feet = feet_ids[gait[t] == 1]
            feet_pos = (
                self.get_active_feet(footsteps[t], support_feet)
                if footsteps is not None
                else ()
            )
            base_pose = base_refs[t] if base_refs is not None else None
            has_switched = np.any(gait[t] != gait[t - 1])
            switch_matrix = gait[t] if has_switched else np.array([])
            switch_feet = feet_ids[switch_matrix == 1]
            running_models.append(
                self.make_running_model(support_feet, switch_feet, feet_pos, base_pose)
            )

        support_feet = feet_ids[gait[-1] == 1]
        terminal_model = self.make_terminal_model(support_feet)

        return running_models, terminal_model

    def solve(self, k):
        t_start = time()

        t_update = time()
        self.t_update = t_update - t_start

        if self.xs_init is None or self.us_init is None:
            self._xs_init = [self.x0] * (self.ddp.problem.T + 1)
            self._us_init = self.ddp.problem.quasiStatic([self.x0] * self.ddp.problem.T)
        self._check_ws_dim()

        t_warm_start = time()
        self.t_warm_start = t_warm_start - t_update

        self.ddp.solve(
            self.xs_init,
            self.us_init,
            self.max_iter if k > 0 else self.init_max_iters,
            False,
        )

        t_ddp = time()
        self.t_ddp = t_ddp - t_warm_start

        self.t_solve = time() - t_start
        self.num_iters = self.ddp.iter

    def make_ocp(self, k, x0, footsteps, base_pose):
        """
        Create a shooting problem for a simple walking gait.

        :param k: current MPC iteration
        :param x0: initial condition
        :param footstep: 2D array
        :param base_pose: 1D array
        """
        self.x0 = x0
        pin.forwardKinematics(self.task.model, self.rdata, self.x0[: self.task.nq])
        pin.updateFramePlacements(self.task.model, self.rdata)

        t = int(k / self.params.mpc_wbc_ratio) - 1

        self.problem.x0 = self.x0
        feet_ids = np.asarray(self.task.feet_ids)

        if k == 0:
            return

        if t < len(self.start_rm):
            mask = self.life_gait[t] == 1
            support_feet = feet_ids[mask]
            model = self.life_rm[t]
            no_copy_roll_insert(self.current_gait, self.life_gait[t])

        elif t < len(self.start_rm) + len(self.life_rm) * self.params.gait_repetitions:
            no_copy_roll(self.life_gait)
            mask = self.life_gait[-1] == 1
            support_feet = feet_ids[mask]
            model = self.problem.runningModels[0]
            no_copy_roll_insert(self.current_gait, self.life_gait[-1])

        else:
            i = (
                0
                if t
                == len(self.start_rm) + len(self.life_rm) * self.params.gait_repetitions
                else 1
            )
            # choose to pich the node with impact or not
            support_feet = feet_ids[self.ending_gait[i] == 1]
            model = self.end_rm[i]
            no_copy_roll_insert(self.current_gait, self.ending_gait[i])
            base_pose = None

        feet_pos = self.get_active_feet(footsteps, support_feet)
        self._update_model(model, feet_pos, base_pose, support_feet)
        self.circular_append(model)
        if k > 0:
            self.cycle_warm_start()

    def circular_append(self, m):
        d = m.createData()
        self.problem.circularAppend(m, d)

    def get_results(self, window_size=None):
        self.xs_init[:] = self.ddp.xs
        self.us_init[:] = self.ddp.us
        if window_size is None:
            window_size = len(self.ddp.us)
        return (
            self.current_gait,
            self.ddp.xs[: window_size + 1],
            self.ddp.us[:window_size],
            self.ddp.K[:window_size].tolist(),
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
        return [m.differential.xout for m in self.ddp.problem.runningDatas]

    def _update_model(
        self,
        model,
        feet_pos,
        base_pose: Optional[np.ndarray],
        support_feet,
        is_terminal=False,
    ):
        for i in self.task.feet_ids:
            name = self.task.model.frames[i].name + "_contact"
            model.differential.contacts.changeContactStatus(name, i in support_feet)
        if not is_terminal:
            self.update_tracking_costs(
                model.differential.costs, feet_pos, base_pose, support_feet
            )

    def _create_standard_model(
        self, support_feet
    ) -> crocoddyl.IntegratedActionModelAbstract:
        """
        Create a standard integrated action model, to be modified by the callee.

        :param state: swinging foot task
        :param support_feet: list of support feet ids
        :return action model for a swing foot phase
        """
        pin.forwardKinematics(self.task.model, self.rdata, self.task.q0)
        pin.updateFramePlacements(self.task.model, self.rdata)
        actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        nu = actuation.nu

        control = crocoddyl.ControlParametrizationModelPolyZero(nu)
        zero_vec = np.zeros(3)

        contacts = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in self.task.feet_ids:
            name = self.task.model.frames[i].name + "_contact"
            contact = crocoddyl.ContactModel3D(
                self.state,
                i,
                zero_vec,
                pin.LOCAL_WORLD_ALIGNED,
                nu,
                self.task.baumgarte_gains,
            )
            contacts.addContact(name, contact)
            contacts.changeContactStatus(name, i in support_feet)

        costs = crocoddyl.CostModelSum(self.state, nu)
        residual = crocoddyl.ResidualModelState(self.state, self.task.xref, nu)
        activation = crocoddyl.ActivationModelWeightedQuad(self.task.state_reg_w**2)
        state_cost = crocoddyl.CostModelResidual(self.state, activation, residual)
        costs.addCost("state_reg", state_cost, 1)

        state_bound_residual = crocoddyl.ResidualModelState(
            self.state, self.task.xref, nu
        )
        activation = crocoddyl.ActivationModelWeightedQuadraticBarrier(
            crocoddyl.ActivationBounds(-self.task.state_limit, self.task.state_limit),
            self.task.state_bound_w**2,
        )
        state_bound_cost = crocoddyl.CostModelResidual(
            self.state, activation, state_bound_residual
        )
        costs.addCost("state_limitBound", state_bound_cost, 1)

        differential = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, actuation, contacts, costs, 0.0, True
        )
        return crocoddyl.IntegratedActionModelEuler(
            differential, control, self.params.dt_mpc
        )

    def make_terminal_model(self, support_feet):
        """
        Add the final velocity cost to the terminal model
        """
        model = self._create_standard_model(support_feet)
        nu = model.differential.actuation.nu
        residual = crocoddyl.ResidualModelState(self.state, self.task.xref, nu)
        activation = crocoddyl.ActivationModelWeightedQuad(
            self.task.terminal_velocity_w**2
        )
        state_cost = crocoddyl.CostModelResidual(self.state, activation, residual)
        model.differential.costs.addCost("terminal_velocity", state_cost, 1)
        return model

    def make_running_model(
        self, support_feet, switch_feet, feet_pos, base_pose: Optional[np.ndarray]
    ):
        """
        Add all the costs to the running models
        """
        model = self._create_standard_model(support_feet)
        nu = model.differential.actuation.nu
        costs = model.differential.costs
        for i in self.task.feet_ids:
            start_pos = self.rdata.oMf[i].translation

            # Contact forces
            cone = crocoddyl.FrictionCone(self.task.Rsurf, self.task.mu, 4, False, 3)
            residual = crocoddyl.ResidualModelContactFrictionCone(
                self.state, i, cone, nu
            )
            activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            friction_cone = crocoddyl.CostModelResidual(
                self.state, activation, residual
            )
            friction_name = self.task.model.frames[i].name + "_friction_cone"
            costs.addCost(friction_name, friction_cone, self.task.friction_cone_w)
            costs.changeCostStatus(friction_name, i in support_feet)

            name = "{}_forceReg".format(self.task.model.frames[i].name)
            nc = len(model.differential.contacts.active_set)
            ref_force = np.array([0, 0, self.task.robot_weight / nc])
            ref_Force = pin.Force(ref_force, ref_force * 0)
            forceRegResidual = crocoddyl.ResidualModelContactForce(
                self.state, i, ref_Force, 3, nu
            )
            forceRegCost = crocoddyl.CostModelResidual(self.state, forceRegResidual)
            costs.addCost(name, forceRegCost, self.task.force_reg_w)
            costs.changeCostStatus(name, False)

            # Tracking foot trajectory
            name = self.task.model.frames[i].name + "_foot_tracking"
            residual = crocoddyl.ResidualModelFrameTranslation(
                self.state, i, np.zeros(3), nu
            )
            foot_tracking = crocoddyl.CostModelResidual(self.state, residual)
            if self.task.foot_tracking_w > 0:
                costs.addCost(name, foot_tracking, self.task.foot_tracking_w)
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

            name = "{}_groundCol".format(self.task.model.frames[i].name)
            if self.task.ground_collision_w > 0:
                costs.addCost(
                    name,
                    groundColCost,
                    self.task.ground_collision_w,
                )
            costs.changeCostStatus(name, False)

            flyHighResidual = sobec.ResidualModelFlyHigh(
                self.state, i, self.task.fly_high_slope / 2.0, nu
            )
            flyHighCost = crocoddyl.CostModelResidual(self.state, flyHighResidual)

            name = "{}_flyHigh".format(self.task.model.frames[i].name)
            if self.task.fly_high_w > 0:
                costs.addCost(
                    name,
                    flyHighCost,
                    self.task.fly_high_w,
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

            name = "{}_vel_zReg".format(self.task.model.frames[i].name)
            vertical_velocity_reg_cost = crocoddyl.CostModelResidual(
                self.state,
                vertical_velocity_activation,
                vertical_velocity_reg_residual,
            )
            costs.addCost(
                name,
                vertical_velocity_reg_cost,
                self.task.vertical_velocity_reg_w,
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
                if self.task.impact_altitude_w > 0:
                    costs.addCost(
                        "{}_altitudeImpact".format(self.task.model.frames[i].name),
                        impactCost,
                        self.task.impact_altitude_w / self.params.dt_mpc,
                    )

                impactVelResidual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    i,
                    pin.Motion.Zero(),
                    pin.ReferenceFrame.WORLD,
                    nu,
                )
                if self.task.impact_velocity_w > 0:
                    impactVelCost = crocoddyl.CostModelResidual(
                        self.state, impactVelResidual
                    )
                    costs.addCost(
                        "{}_velimpact".format(self.task.model.frames[i].name),
                        impactVelCost,
                        self.task.impact_velocity_w / self.params.dt_mpc,
                    )

        name = "base_velocity_tracking"
        if base_pose is not None:
            ref = pin.Motion(base_pose)
        else:
            ref = pin.Motion.Zero()

        residual_base_velocity = crocoddyl.ResidualModelFrameVelocity(
            self.state, self.task.base_id, ref, pin.LOCAL, nu
        )
        base_velocity = crocoddyl.CostModelResidual(self.state, residual_base_velocity)

        if self.task.base_velocity_tracking_w > 0:
            costs.addCost(name, base_velocity, self.task.base_velocity_tracking_w)

        control_residual = crocoddyl.ResidualModelControl(self.state, self.task.uref)
        control_reg = crocoddyl.CostModelResidual(self.state, control_residual)
        costs.addCost("control_reg", control_reg, self.task.control_reg_w)

        control_bound_residual = crocoddyl.ResidualModelControl(self.state, nu)
        control_bound_activation = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(-self.task.effort_limit, self.task.effort_limit)
        )
        control_bound = crocoddyl.CostModelResidual(
            self.state, control_bound_activation, control_bound_residual
        )
        costs.addCost("control_bound", control_bound, self.task.control_bound_w)

        self.update_tracking_costs(costs, feet_pos, base_pose, support_feet)
        return model

    def update_tracking_costs(
        self, costs, feet_pos, base_pose: Optional[np.ndarray], support_feet
    ):
        index = 0
        for i in self.task.feet_ids:
            if self.task.foot_tracking_w > 0:
                name = "{}_foot_tracking".format(self.task.model.frames[i].name)
                if i in feet_pos[0]:
                    costs.costs[name].cost.residual.reference = feet_pos[1][index]
                    index += 1
                costs.changeCostStatus(name, i not in support_feet)

            name = "{}_forceReg".format(self.task.model.frames[i].name)
            costs.changeCostStatus(name, i in support_feet)

            name = "{}_groundCol".format(self.task.model.frames[i].name)
            costs.changeCostStatus(name, i not in support_feet)

            name = "{}_flyHigh".format(self.task.model.frames[i].name)
            costs.changeCostStatus(name, i not in support_feet)

            name = "{}_vel_zReg".format(self.task.model.frames[i].name)
            costs.changeCostStatus(name, i not in support_feet)

        if base_pose is not None and self.task.base_velocity_tracking_w > 0:
            name = "base_velocity_tracking"
            costs.costs[name].cost.residual.reference.np[:] = base_pose[:6]
