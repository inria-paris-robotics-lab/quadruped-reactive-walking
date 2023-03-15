import numpy as np
import quadruped_reactive_walking as qrw

from quadruped_reactive_walking.wb_mpc import AlgtrOCP, CrocOCP
from quadruped_reactive_walking.controller import make_footsteps_and_refs
from quadruped_reactive_walking.wb_mpc.task_spec import TaskSpec
from quadruped_reactive_walking.wb_mpc.target import Target
from quadruped_reactive_walking.tools.Utils import init_robot

params = qrw.Params()
params.ocp: qrw.OCPParams
params.ocp.init_max_iters = 1
q_init = np.array(params.q_init.tolist())
init_robot(q_init, params)
task = TaskSpec(params)

print(task)

target = Target(params)

footsteps, base_refs = make_footsteps_and_refs(params, target)

x0 = task.x0
print("x0:", x0)

ocp = CrocOCP(params, footsteps, base_refs)

nsteps = ocp.ddp.problem.T
xs_i = [x0] * (nsteps + 1)
us_i = ocp.problem.quasiStatic(xs_i[:nsteps])

ocp.solve(0, xs_i, us_i)


ocp2 = AlgtrOCP(params, footsteps, base_refs)
ocp2.solve(0, xs_i, us_i)
ocp2_res = ocp2.prox_ddp.getResults()


dist_x = np.linalg.norm(np.stack(ocp.ddp.xs) - np.stack(ocp2_res.xs))
print("Dist X:", dist_x)
