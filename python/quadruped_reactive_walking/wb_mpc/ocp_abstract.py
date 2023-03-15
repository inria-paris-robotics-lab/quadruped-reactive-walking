import abc

from .problem_data import TaskSpec
import quadruped_reactive_walking as qrw


class OCPAbstract(abc.ABC):
    num_iters: int

    def __init__(self, params: qrw.Params):
        self.task = TaskSpec(params)
        self.params = params
        self.max_iter = 1000 if params.save_guess else params.ocp.max_iter
        self.init_max_iters = params.ocp.init_max_iters

    @abc.abstractmethod
    def initialize_models_from_gait(self, gait, footsteps=None, base_refs=None):
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
        """Get the positions for all the active feet."""
        feet_pos = ([], [])
        for i, fid in enumerate(self.task.feet_ids):
            if fid in support_feet:
                feet_pos[0].append(fid)
                feet_pos[1].append(footstep[:, i])
        return feet_pos
