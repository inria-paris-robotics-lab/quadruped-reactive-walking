import abc

from .task_spec import TaskSpec
import quadruped_reactive_walking as qrw


class _OCPMeta(type(qrw.IOCPAbstract), abc.ABCMeta):
    pass


class OCPAbstract(qrw.IOCPAbstract, metaclass=_OCPMeta):
    def __init__(self, params: qrw.Params):
        super().__init__(params)
        self.task = TaskSpec(params)

    @abc.abstractmethod
    def initialize_models_from_gait(self, gait, footsteps=None, base_refs=None):
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
