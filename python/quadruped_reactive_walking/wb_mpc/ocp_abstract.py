import abc

import quadruped_reactive_walking as qrw


class _OCPMeta(type(qrw.IOCPAbstract), abc.ABCMeta):
    pass


class OCPAbstract(qrw.IOCPAbstract, metaclass=_OCPMeta):
    def __init__(self, params: qrw.Params):
        super().__init__(params)

    @abc.abstractstaticmethod
    def get_type_str():
        pass

    @abc.abstractclassmethod
    def circular_append(self, model):
        pass
