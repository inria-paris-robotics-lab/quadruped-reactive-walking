import numpy as np
import quadruped_reactive_walking as qrw
import time

from quadruped_reactive_walking.wb_mpc import AlgtrOCPProx, CrocOCP
from quadruped_reactive_walking.controller import make_footsteps_and_refs
from quadruped_reactive_walking.wb_mpc.task_spec import TaskSpec
from quadruped_reactive_walking.wb_mpc.target import Target

params = qrw.Params.create_from_file()
params.ocp: qrw.OCPParams
params.ocp.init_max_iters = 1
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


ocp2 = AlgtrOCPProx(params, footsteps, base_refs)

ts = time.time()
n = 2000
for i in range(n):
    ocp2.solve(0, xs_i, us_i)
elapsed = time.time() - ts
print("Elapsed time: {}".format(elapsed))
print("Avg. time   : {}".format(elapsed / n))

ocp2_res = ocp2.prox_ddp.getResults()


dist_x = np.linalg.norm(np.stack(ocp.ddp.xs) - np.stack(ocp2_res.xs))
print("Dist X:", dist_x)
