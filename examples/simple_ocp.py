import numpy as np
import quadruped_reactive_walking as qrw
from quadruped_reactive_walking.controller import Controller
import time
import pprint

from quadruped_reactive_walking.wb_mpc import AlgtrOCPProx, CrocOCP, OCP_TYPE_MAP
from quadruped_reactive_walking.controller import make_footsteps_and_refs
from quadruped_reactive_walking.wb_mpc.task_spec import TaskSpec
from quadruped_reactive_walking.wb_mpc.target import Target

print("OCP registered types:", pprint.pformat(OCP_TYPE_MAP), sep="\n")
params = qrw.Params.create_from_file()
params.ocp: qrw.OCPParams
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

ocp.xs_init = xs_i
ocp.us_init = us_i
ocp.solve(0)


ocp2 = AlgtrOCPProx(params, footsteps, base_refs)

ts = time.time()
n = 1
for i in range(n):
    ocp2.solve(0)
elapsed = time.time() - ts
print("Elapsed time: {}".format(elapsed))
print("Avg. time   : {}".format(elapsed / n))

ocp2_res = ocp2.prox_ddp.results

dist_x = np.linalg.norm(np.stack(ocp.ddp.xs) - np.stack(ocp2_res.xs))
print("Dist X:", dist_x)

ctrler = Controller(params, params.q_init, CrocOCP)
