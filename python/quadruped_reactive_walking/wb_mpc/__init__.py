from .ocp_abstract import OCPAbstract
from .ocp_crocoddyl import CrocOCP
from .ocp_proxddp import AlgtrOCPProx, AlgtrOCPFDDP

_OCP_TYPES = (CrocOCP, AlgtrOCPProx, AlgtrOCPFDDP)


def get_ocp_from_str(type_str):
    for ocp in _OCP_TYPES:
        if ocp.get_type_str() == type_str:
            return ocp
    raise ValueError("No OCP class named: " + type_str)


def get_ocp_list_str():
    return [ocp.get_type_str() for ocp in _OCP_TYPES]


OCP_TYPE_MAP = {ocp_cls.get_type_str(): ocp_cls for ocp_cls in _OCP_TYPES}


__all__ = ["OCPAbstract", "CrocOCP", "AlgtrOCPProx", "AlgtrOCPFDDP", "OCP_TYPE_MAP"]
