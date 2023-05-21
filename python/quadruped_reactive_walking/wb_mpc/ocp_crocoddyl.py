import crocoddyl
import pinocchio as pin
import numpy as np
import time

from colorama import Fore
from crocoddyl import StateMultibody
from .ocp_abstract import OCPAbstract
from typing import Optional
from ..tools.utils import no_copy_roll, no_copy_roll_insert
from quadruped_reactive_walking import Params
from . import task_spec
from .walking import WalkingOCPBuilder, get_active_feet


class CrocOCP(OCPAbstract):
    """
    Generate a Crocoddyl OCP for the control task.
    """

    def __init__(self, params: Params, footsteps, base_refs):
        super().__init__(params)
        self.task = task_spec.TaskSpec(params)
        self.state = StateMultibody(self.rmodel)
        self.rdata = self.task.model.createData()

        self._builder = WalkingOCPBuilder(params)

        # Set the problem parameters
        self.t_problem_update = 0
        self.t_update_last_model = 0.0
        self.t_shift = 0.0

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
        if self.warm_start_empty():
            print(Fore.CYAN + "No warm-start found, initializing..." + Fore.RESET)
            self.xs_init = [self.x0] * (self.ddp.problem.T + 1)
            self.us_init = self.ddp.problem.quasiStatic([self.x0] * self.ddp.problem.T)

    @property
    def rmodel(self):
        return self.task.model

    def get_type_str():
        return "croc"

    def initialize_models_from_gait(self, gait, footsteps=None, base_vel_refs=None):
        """Create action models (problem stages) from a gait matrix and other optional data."""
        # both or neither must be none
        assert (footsteps is None) == (base_vel_refs is None)
        if footsteps is not None:
            assert len(footsteps) == len(base_vel_refs)
        running_models = []
        feet_ids = np.asarray(self.task.feet_ids)
        for t in range(gait.shape[0]):
            support_feet_ids = feet_ids[gait[t] == 1]
            feet_pos = (
                get_active_feet(footsteps[t], support_feet_ids)
                if footsteps is not None
                else []
            )
            base_vel_ref = base_vel_refs[t] if base_vel_refs is not None else None
            has_switched = np.any(gait[t] != gait[t - 1])
            switch_matrix = gait[t] if has_switched else np.array([])
            switch_feet = feet_ids[switch_matrix == 1]
            running_models.append(
                self._builder.make_running_model(
                    support_feet_ids, switch_feet, feet_pos, base_vel_ref
                )
            )

        support_feet_ids = feet_ids[gait[-1] == 1]
        terminal_model = self._builder.make_terminal_model(support_feet_ids)

        return running_models, terminal_model

    def solve(self, k):
        t_start = time.time()

        t_update = time.time()
        self.t_update = t_update - t_start

        self._check_ws_dim()

        t_warm_start = time.time()
        self.t_warm_start = t_warm_start - t_update

        self.ddp.solve(
            self.xs_init,
            self.us_init,
            self.max_iter if k > 0 else self.init_max_iters,
            False,
        )

        t_ddp = time.time()
        self.t_ddp = t_ddp - t_warm_start

        self.t_solve = time.time() - t_start
        self.num_iters = self.ddp.iter

    def push_node(self, k, x0, footsteps, base_vel_ref: Optional[pin.Motion]):
        """
        Create a shooting problem for a simple walking gait.

        :param k: current MPC iteration
        :param x0: initial condition
        :param footstep: 2D array
        :param base_pose: 1D array
        """
        self.x0 = x0
        pin.forwardKinematics(self.rmodel, self.rdata, self.x0[: self.task.nq])
        pin.updateFramePlacements(self.rmodel, self.rdata)

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
            base_vel_ref = None

        if base_vel_ref is not None:
            base_vel_ref = pin.Motion(base_vel_ref)

        active_feet_pos = get_active_feet(footsteps, support_feet)
        self._builder.update_model(model, active_feet_pos, base_vel_ref, support_feet)
        self.circular_append(model)
        self.cycle_warm_start()

    def circular_append(self, m):
        d = m.createData()
        self.problem.circularAppend(m, d)

    def get_results(self, window_size=None):
        self.xs_init = self.ddp.xs
        self.us_init = self.ddp.us
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
