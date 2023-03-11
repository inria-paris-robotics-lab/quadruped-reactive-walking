import abc

from .problem_data import TaskSpec
import quadruped_reactive_walking as qrw


class OCPAbstract(abc.ABC):
    num_iters: int

    def __init__(self, params: qrw.Params):
        self.pd = TaskSpec(params)
        self.params = params
        self.max_iter = 1000 if params.save_guess else params.ocp.max_iter
        self.init_max_iters = params.ocp.init_max_iters

    @abc.abstractmethod
    def initialize_models(self, gait, footsteps=[], base_refs=[]):
        pass

    @abc.abstractmethod
    def solve(self, k, xs_init, us_init):
        pass

    @abc.abstractmethod
    def get_results(self) -> tuple:
        pass

    @abc.abstractmethod
    def make_ocp(self, k, x0, footstep, base_task):
        pass

    def get_active_feet(self, footstep):
        task = ([], [])
        for foot in range(4):
            if footstep[:, foot].any():
                task[0].append(self.pd.feet_ids[foot])
                task[1].append(footstep[:, foot])
        return task
