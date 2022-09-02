from tracemalloc import take_snapshot
import numpy as np
from .ProblemData import ProblemData
import pinocchio as pin
from scipy.interpolate import KroghInterpolator


class Target:
    def __init__(self, params, foot_pose):
        self.params = params
        self.dt_wbc = params.dt_wbc
        self.k_per_step = 160

        if params.movement == "base_circle":
            self.initial_base = np.array([0.0, 0.0, params.h_ref])
            self.A = np.array([0.02, 0.0, 0.0])
            self.offset = np.array([0.0, 0.0, 0.0])
            self.freq = np.array([0.5, 0.5, 0.0])
            self.phase = np.array([0.0, 0.0, 0.0])
        elif params.movement == "circle":
            self.position = np.array(params.footsteps_init.tolist()).reshape(
                (3, 4), order="F"
            )
            self.A = np.array([0.05, 0.0, 0.04])
            self.offset = np.array([0.05, 0, 0.05])
            self.freq = np.array([0.5, 0.0, 0.5])
            self.phase = np.array([-np.pi / 2 - 0.5, 0.0, -np.pi / 2])
        elif params.movement == "step":
            self.p0 = foot_pose
            self.p1 = foot_pose.copy() + np.array([0.025, 0.0, 0.03])
            self.v1 = np.array([0.5, 0.0, 0.0])
            self.p2 = foot_pose.copy() + np.array([0.05, 0.0, 0.0])

            self.T = self.k_per_step * self.dt_wbc
            self.ts = np.repeat(np.linspace(0, self.T, 3), 2)

            self.update_time = -1
        else:
            self.target_footstep = np.array(
                self.params.footsteps_init.tolist()
            ).reshape((3, 4), order="F")
            self.ramp_length = 100
            self.target_ramp_x = np.linspace(0.0, -0.0, self.ramp_length)
            self.target_ramp_y = np.linspace(0.0, 0.0, self.ramp_length)
            self.target_ramp_z = np.linspace(0.0, 0.05, self.ramp_length)

    def compute(self, k):
        footstep = np.zeros((3, 4))
        if self.params.movement == "base_circle":
            target = self.evaluate_circle(k, self.initial_base)
        elif self.params.movement == "circle":
            target[:, 1] = self.evaluate_circle(k, self.position[:, 1])
        elif self.params.movement == "step":
            target[:, 1] = self.evaluate_step(1, k)
            target[2, 1] += 0.015
        else:
            target = self.target_footstep.copy()
            target[0, 1] = (
                self.target_ramp_x[k]
                if k < self.ramp_length
                else self.target_ramp_x[-1]
            )
            target[1, 1] = (
                self.target_ramp_y[k]
                if k < self.ramp_length
                else self.target_ramp_y[-1]
            )
            target[2, 1] = (
                self.target_ramp_z[k]
                if k < self.ramp_length
                else self.target_ramp_z[-1]
            )

        return target

    def evaluate_circle(self, k, initial_position):
        return (
            initial_position
            + self.offset
            + self.A * np.sin(2 * np.pi * self.freq * k * self.dt_wbc + self.phase)
        )

    def evaluate_step(self, j, k):
        n_step = k // self.k_per_step
        if n_step % 2 == 0:
            return self.p0.copy() if (n_step % 4 == 0) else self.p2.copy()

        if n_step % 4 == 1:
            initial = self.p0
            target = self.p2
            velocity = self.v1
        else:
            initial = self.p2
            target = self.p0
            velocity = -self.v1

        k_step = k % self.k_per_step
        if n_step != self.update_time:
            self.update_interpolator(initial, target, velocity)
            self.update_time = n_step

        t = k_step * self.dt_wbc
        return self.compute_position(t)

    def update_interpolator(self, initial, target, velocity):
        self.y = [initial, np.zeros(3), self.p1, velocity, target, np.zeros(3)]
        self.krog = KroghInterpolator(self.ts, np.array(self.y))

    def compute_position(self, t):
        p = self.krog(t)
        # v = self.krog.derivative(t)
        return p
