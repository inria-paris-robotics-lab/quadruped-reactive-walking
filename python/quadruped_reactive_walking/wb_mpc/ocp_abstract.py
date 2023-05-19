import abc
import numpy as np

from typing import List
import quadruped_reactive_walking as qrw


class _OCPMeta(type(qrw.IOCPAbstract), abc.ABCMeta):
    pass


class OCPAbstract(qrw.IOCPAbstract, metaclass=_OCPMeta):
    def __init__(self, params: qrw.Params):
        super().__init__(params)

    @abc.abstractmethod
    def initialize_models_from_gait(self, gait, footsteps=None, base_refs=None):
        pass

    @abc.abstractstaticmethod
    def get_type_str():
        pass

    @abc.abstractclassmethod
    def circular_append(self, model):
        pass

    def get_active_feet(self, footstep, support_feet) -> List[np.ndarray]:
        """Get the positions for all the active feet."""
        feet_pos = []
        for i, fid in enumerate(support_feet):
            feet_pos.append(footstep[:, i])
        return feet_pos
