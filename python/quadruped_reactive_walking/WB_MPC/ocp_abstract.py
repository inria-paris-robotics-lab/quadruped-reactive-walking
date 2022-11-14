import abc
import numpy as np

from .problem_data import ProblemData
import quadruped_reactive_walking as qrw


class OCPAbstract(abc.ABC):
    def __init__(self, pd: ProblemData, params: qrw.Params):
        self.pd = pd
        self.params = params
        self.max_iter = 1000 if params.save_guess else 1

    def _init_impl(self, footsteps, base_refs):
        """Set the problem parameters."""

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

    @property
    def rmodel(self):
        return self.pd.model

    @abc.abstractmethod
    def initialize_models(self, gait, footsteps=[], base_refs=[]):
        pass

    @abc.abstractmethod
    def solve(self, k, x0, footstep, base_ref, xs_init, us_init):
        pass

    @abc.abstractmethod
    def get_results(self):
        pass

    def make_task(self, footstep):
        task = [[], []]
        # if base_ref is not None:
        #     task[0].append(self.pd.base_id)
        #     task[1].append(base_ref)
        for foot in range(4):
            if footstep[:, foot].any():
                task[0].append(self.pd.feet_ids[foot])
                task[1].append(footstep[:, foot])
        return task
