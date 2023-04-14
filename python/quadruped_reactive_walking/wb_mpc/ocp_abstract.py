import abc

from .task_spec import TaskSpec
import quadruped_reactive_walking as qrw
from ..tools.utils import no_copy_roll_insert


class OCPAbstract(abc.ABC):
    num_iters: int

    def __init__(self, params: qrw.Params):
        self.task = TaskSpec(params)
        self.params = params
        self.max_iter = 1000 if params.save_guess else params.ocp.max_iter
        self.init_max_iters = params.ocp.init_max_iters

        # Warm starts. To be stored internally.
        # Should not be set from the outside
        self._xs_init = None
        self._us_init = None

    @property
    def xs_init(self):
        return self._xs_init

    @property
    def us_init(self):
        return self._us_init

    def cycle_warm_start(self):
        xT = self.xs_init[-1]
        no_copy_roll_insert(self.xs_init, xT)

        uT = self.us_init[-1]
        no_copy_roll_insert(self.us_init, uT)

    def _check_ws_dim(self):
        """
        Check size of warmstart buffers.
        Must be called in solve()
        """
        N = self.params.N_gait
        assert len(self.xs_init) == N + 1
        assert len(self.us_init) == N

    @abc.abstractmethod
    def initialize_models_from_gait(self, gait, footsteps=None, base_refs=None):
        pass

    @abc.abstractmethod
    def solve(self, k, xs_init, us_init):
        pass

    @abc.abstractmethod
    def get_results(self, window_size=None) -> tuple:
        pass

    @abc.abstractmethod
    def make_ocp(self, k, x0, footstep, base_task):
        pass

    @abc.abstractstaticmethod
    def get_type_str():
        pass

    def get_active_feet(self, footstep, support_feet):
        """Get the positions for all the active feet."""
        feet_pos = ([], [])
        for i, fid in enumerate(self.task.feet_ids):
            if fid in support_feet:
                feet_pos[0].append(fid)
                feet_pos[1].append(footstep[:, i])
        return feet_pos
