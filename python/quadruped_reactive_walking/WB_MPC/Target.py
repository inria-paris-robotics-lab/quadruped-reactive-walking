from tracemalloc import take_snapshot
import numpy as np
from .ProblemData import ProblemData
import pinocchio as pin


class Target:
    def __init__(self, params):
        self.params = params
        self.dt_wbc = params.dt_wbc
        self.k_per_step = 160

        self.position = np.array(params.footsteps_under_shoulders).reshape(
            (3, 4), order="F"
        )

        if params.movement == "circle":
            self.A = np.array([0, 0.03, 0.03])
            self.offset = np.array([0.05, -0.02, 0.06])
            self.freq = np.array([0, 0.5, 0.5])
            self.phase = np.array([0, np.pi / 2, 0])
        elif params.movement == "step":
            self.initial = self.position[:, 1].copy()
            self.target = self.position[:, 1].copy() + np.array([0.1, 0.0, 0.0])

            self.vert_time = params.vert_time
            self.max_height = params.max_height
            self.T = self.k_per_step * self.dt_wbc
            self.A = np.zeros((6, 3))

            self.update_time = -1
        else:
            self.target_footstep = self.position + np.array([0.0, 0.0, 0.10])

    def compute(self, k):
        footstep = np.zeros((3, 4))
        if self.params.movement == "circle":
            footstep[:, 1] = self.evaluate_circle(k)
        elif self.params.movement == "step":
            footstep[:, 1] = self.evaluate_step(1, k)
        else:
            footstep = self.target_footstep.copy()

        return footstep

    def evaluate_circle(self, k):
        return (
            self.position[:, 1]
            + self.offset
            + self.A * np.sin(2 * np.pi * self.freq * k * self.dt_wbc + self.phase)
        )

    def evaluate_step(self, j, k):
        n_step = k // self.k_per_step
        if n_step % 2 == 0:
            return self.initial.copy() if (n_step % 4 == 0) else self.target.copy()

        if n_step % 4 == 1:
            initial = self.initial
            target = self.target
        else:
            initial = self.target
            target = self.initial

        k_step = k % self.k_per_step
        if n_step != self.update_time:
            self.update_polynomial(initial, target)
            self.update_time = n_step

        t = k_step * self.dt_wbc
        return self.compute_position(j, t)

    def update_polynomial(self, initial, target):

        x0 = initial[0]
        y0 = initial[1]

        x1 = target[0]
        y1 = target[1]

        # elapsed time
        t = 0
        d = self.T - 2 * self.vert_time

        A = np.zeros((6, 3))

        A[0, 0] = 12 * (x0 - x1) / (2 * (t - d) ** 5)
        A[1, 0] = 30 * (x1 - x0) * (t + d) / (2 * (t - d) ** 5)
        A[2, 0] = 20 * (x0 - x1) * (t**2 + d**2 + 4 * t * d) / (2 * (t - d) ** 5)
        A[3, 0] = 60 * (x1 - x0) * (t * d**2 + t**2 * d) / (2 * (t - d) ** 5)
        A[4, 0] = 60 * (x0 - x1) * (t**2 * d**2) / (2 * (t - d) ** 5)
        A[5, 0] = (
            2 * x1 * t**5
            - 10 * x1 * t**4 * d
            + 20 * x1 * t**3 * d**2
            - 20 * x0 * t**2 * d**3
            + 10 * x0 * t * d**4
            - 2 * x0 * d**5
        ) / (2 * (t - d) ** 5)

        A[0, 1] = 12 * (y0 - y1) / (2 * (t - d) ** 5)
        A[1, 1] = 30 * (y1 - y0) * (t + d) / (2 * (t - d) ** 5)
        A[2, 1] = 20 * (y0 - y1) * (t**2 + d**2 + 4 * t * d) / (2 * (t - d) ** 5)
        A[3, 1] = 60 * (y1 - y0) * (t * d**2 + t**2 * d) / (2 * (t - d) ** 5)
        A[4, 1] = 60 * (y0 - y1) * (t**2 * d**2) / (2 * (t - d) ** 5)
        A[5, 1] = (
            2 * y1 * t**5
            - 10 * y1 * t**4 * d
            + 20 * y1 * t**3 * d**2
            - 20 * y0 * t**2 * d**3
            + 10 * y0 * t * d**4
            - 2 * y0 * d**5
        ) / (2 * (t - d) ** 5)

        A[0, 2] = -self.max_height / ((self.T / 2) ** 6)
        A[1, 2] = 3 * self.T * self.max_height / ((self.T / 2) ** 6)
        A[2, 2] = -3 * self.T**2 * self.max_height / ((self.T / 2) ** 6)
        A[3, 2] = self.T**3 * self.max_height / ((self.T / 2) ** 6)

        self.A = A

    def compute_position(self, j, t):
        A = self.A.copy()

        t_xy = t - self.vert_time
        t_xy = max(0.0, t_xy)
        t_xy = min(t_xy, self.T - 2 * self.vert_time)
        self.position[:2, j] = (
            A[5, :2]
            + A[4, :2] * t_xy
            + A[3, :2] * t_xy**2
            + A[2, :2] * t_xy**3
            + A[1, :2] * t_xy**4
            + A[0, :2] * t_xy**5
        )

        self.position[2, j] = (
            A[3, 2] * t**3 + A[2, 2] * t**4 + A[1, 2] * t**5 + A[0, 2] * t**6
        )

        return self.position[:, j]
