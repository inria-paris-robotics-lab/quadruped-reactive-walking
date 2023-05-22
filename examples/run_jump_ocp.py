import pprint

from quadruped_reactive_walking import Params
from quadruped_reactive_walking.ocp_defs import jump
from quadruped_reactive_walking.wb_mpc.target import Target, make_footsteps_and_refs


params = Params.create_from_file()
target = Target(params)
footsteps, base_refs = make_footsteps_and_refs(params, target)

ocp_spec = jump.JumpOCPBuilder(params, footsteps, base_refs)

pprint.pprint(ocp_spec.add_spec)
