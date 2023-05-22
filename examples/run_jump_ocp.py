import pprint
import crocoddyl
import numpy as np
import matplotlib.pyplot as plt

from quadruped_reactive_walking import Params
from quadruped_reactive_walking.ocp_defs import jump
from quadruped_reactive_walking.wb_mpc.target import Target, make_footsteps_and_refs
from crocoddyl import SolverFDDP, ShootingProblem
from pinocchio.visualize import MeshcatVisualizer
from quadruped_reactive_walking.tools.kinematics_utils import get_translation_array


params = Params.create_from_file()
target = Target(params)
footsteps, base_vel_refs = make_footsteps_and_refs(params, target)

ocp_spec = jump.JumpOCPBuilder(params, footsteps, base_vel_refs)
robot = ocp_spec.task.robot
rmodel = robot.model

pprint.pprint(ocp_spec.jump_spec)

problem: ShootingProblem = ocp_spec.problem
solver = SolverFDDP(problem)
nsteps = params.N_gait
dt = params.dt_mpc
x0 = ocp_spec.x0
xs_init = [x0 for _ in range(params.N_gait + 1)]
us_init = problem.quasiStatic(xs_init[:nsteps])


def make_meshcat_viz():
    import hppfcl
    import pinocchio as pin
    import numpy as np

    plane = hppfcl.Plane(np.array([0, 0, 1]), 0.0)
    geobj = pin.GeometryObject("plane", 0, pin.SE3.Identity(), plane)
    geobj.meshColor[:] = 0.6, 0.1, 0.1, 0.8
    geobj.meshScale *= 2.0

    vmodel = robot.visual_model
    vmodel.addGeometryObject(geobj)

    vizer = MeshcatVisualizer(
        robot.model, robot.collision_model, vmodel, data=robot.data
    )
    vizer.initViewer(loadModel=True)
    vizer.setBackgroundColor()
    return vizer


solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.solve(xs_init, us_init)
xs = solver.xs
us = solver.us
qs = [x[: rmodel.nq] for x in xs]

ROOT_JOINT_ID = rmodel.getFrameId("base_link")
frame_pos, frame_vels = get_translation_array(rmodel, xs, ROOT_JOINT_ID)

times = np.linspace(0.0, nsteps * dt, nsteps + 1)

labs = ["x", "y", "z"]
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(times, frame_pos[:, i])
    plt.ylabel("${}$".format(labs[i]))
plt.tight_layout()
plt.show()


vizer = make_meshcat_viz()
vizer.viewer.open()

input()
vizer.play(qs, dt)
