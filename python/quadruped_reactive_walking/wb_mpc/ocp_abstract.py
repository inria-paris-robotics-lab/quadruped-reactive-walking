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
    def initialize_models_from_gait(self, gait, footsteps=[], base_refs=[]):
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

    def get_active_feet(self, footstep, support_feet):
        # Use the mask to get the subset of footsteps for active feet.
        task = ([], [])
        for i, fid in enumerate(self.pd.feet_ids):
            if fid in support_feet:
                task[0].append(self.pd.feet_ids[i])
                task[1].append(footstep[:, i])
        return task
