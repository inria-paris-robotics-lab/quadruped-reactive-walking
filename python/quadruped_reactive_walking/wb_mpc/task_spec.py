import numpy as np
import example_robot_data as erd
import pinocchio as pin


class TaskSpecBase:
    def __init__(self, params, frozen_names=[]):

        self.robot = erd.load("solo12")
        self.q0 = self.robot.q0
        self.q0[:7] = np.array([0.0, 0.0, params.h_ref, 0, 0, 0, 1])
        self.q0[7:] = params.q_init

        self.model = self.robot.model
        self.collision_model = self.robot.collision_model
        self.visual_model = self.robot.visual_model

        self.robot_weight = (
            -sum([Y.mass for Y in self.model.inertias]) * self.model.gravity.linear[2]
        )

        self.frozen_names = frozen_names
        if frozen_names != []:
            self.frozen_idxs = [self.model.getJointId(id) for id in frozen_names]
            self.freeze()

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.ndx = 2 * self.nv
        # -1 to take into account the freeflyer
        self.nu = 12 - (len(frozen_names) - 1) if len(frozen_names) != 0 else 12

        self.effort_limit = np.ones(self.nu) * 2.5

        self.v0 = np.zeros(18)
        self.x0 = np.concatenate([self.q0, self.v0])

        pin.centerOfMass(self.robot.model, self.robot.data, self.q0, np.zeros((18, 1)))
        self.u0 = np.array([0.0, 0.0, 9.81 * self.robot.data.mass[0] / 4.0] * 4)

        self.baumgarte_gains = np.array([0, 100])

        self.feet_names = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.feet_ids = [self.model.getFrameId(f) for f in self.feet_names]

        self.Rsurf = np.eye(3)

    def create_rdata(self):
        return self.model.createData()

    def freeze(self):
        geom_models = [self.visual_model, self.collision_model]
        self.model, geometric_models_reduced = pin.buildReducedModel(
            self.model,
            list_of_geom_models=geom_models,
            list_of_joints_to_lock=self.frozen_idxs,
            reference_configuration=self.q0,
        )
        self.rdata = self.model.createData()
        self.visual_model = geometric_models_reduced[0]
        self.collision_model = geometric_models_reduced[1]


class TaskSpec(TaskSpecBase):
    def __init__(self, params):
        super().__init__(params)

        self.useFixedBase = 0
        self.base_id = self.model.getFrameId("base_link")

        self.state_limit = np.concatenate(
            [np.full(18, np.inf), np.zeros(6), np.ones(12) * 800]
        )

        # Cost function weights
        self.mu = 0.7

        # if params.movement == "step":
        #     self.foot_tracking_w = 2.0 * 1e3
        # else:
        #     self.foot_tracking_w = 1e4
        # self.base_tracking_w = 0.
        # self.friction_cone_w = 0.0  # 1e4
        # self.control_bound_w = 0.

        self.fly_high_slope = 50
        self.fly_high_w = 5 * 1e4
        self.ground_collision_w = 1e3
        self.vertical_velocity_reg_w = 1e3

        self.base_velocity_tracking_w = 8 * 1e5
        self.foot_tracking_w = 0

        self.impact_altitude_w = 1e6
        self.impact_velocity_w = 1e4
        self.friction_cone_w = 1e3 * 0.0

        self.control_bound_w = 1e4
        self.control_reg_w = 1e4
        self.state_reg_w = np.array(
            [0] * 3 + [0] * 3 + [1e2 * 3] * 12 + [0] * 6 + [1e1 * 2] * 12
        )
        self.state_bound_w = np.array([0] * 18 + [0] * 6 + [0] * 12)
        self.terminal_velocity_w = np.array([0] * self.nv + [1e3] * self.nv)
        self.force_reg_w = 1e2

        self.xref = self.x0
        self.uref = np.array(
            [
                -0.02615051,
                -0.25848605,
                0.51696646,
                0.0285894,
                -0.25720605,
                0.51441775,
                -0.02614404,
                0.25848271,
                -0.51697107,
                0.02859587,
                0.25720939,
                -0.51441314,
            ]
        )


class TaskSpecFull(TaskSpecBase):
    def __init__(self, params):
        frozen_names = ["root_joint"]

        super().__init__(params, frozen_names)

        self.useFixedBase = 1

        # Cost function weights
        self.mu = 0.7
        self.foot_tracking_w = 1e4
        self.friction_cone_w = 1e3
        self.control_bound_w = 1e3
        self.control_reg_w = 1e0
        self.state_reg_w = np.array(
            [1e2] * 3 + [1e-2] * 3 + [1e2] * 6 + [1e1] * 3 + [1e0] * 3 + [1e1] * 6
        )
        self.terminal_velocity_w = np.array([0] * 12 + [1e3] * 12)

        self.q0_reduced = self.q0[7:]
        self.v0_reduced = np.zeros(self.nv)
        self.x0_reduced = np.concatenate([self.q0_reduced, self.v0_reduced])

        self.xref = self.x0_reduced
        self.uref = self.u0
