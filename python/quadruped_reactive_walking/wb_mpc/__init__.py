from .ocp_abstract import OCPAbstract
from .ocp_crocoddyl import CrocOCP
from .ocp_proxddp import AlgtrOCPAbstract, AlgtrOCPProx, AlgtrOCPFDDP

OCP_TYPES = (CrocOCP, AlgtrOCPProx, AlgtrOCPFDDP)


def get_ocp_from_str(type_str):
    for ocp in OCP_TYPES:
        if ocp.get_type_str() == type_str:
            return ocp
    raise ValueError("No OCP class named: " + type_str)


def get_ocp_list_str():
    return [ocp.get_type_str() for ocp in OCP_TYPES]
