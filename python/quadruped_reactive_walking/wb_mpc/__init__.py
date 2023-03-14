from .ocp_abstract import OCPAbstract
from .ocp_crocoddyl import CrocOCP
from .ocp_proxddp import AlgtrOCP

def get_ocp_from_str(type_str):
    for ocp in [CrocOCP, AlgtrOCP]:
        if(ocp.get_type_str == type_str):
            return ocp
    return None