import pprint
import crocoddyl

from quadruped_reactive_walking import Params
from quadruped_reactive_walking.ocp_defs import jump
from quadruped_reactive_walking.wb_mpc.target import Target, make_footsteps_and_refs
from crocoddyl import SolverFDDP, ShootingProblem
from pinocchio.visualize import MeshcatVisualizer


params = Params.create_from_file()
target = Target(params)
footsteps, base_vel_refs = make_footsteps_and_refs(params, target)

ocp_spec = jump.JumpOCPBuilder(params, footsteps, base_vel_refs)
robot = ocp_spec.task.robot

pprint.pprint(ocp_spec.jump_spec)

problem: ShootingProblem = ocp_spec.problem
solver = SolverFDDP(problem)
nsteps = params.N_gait
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


# vizer = make_meshcat_viz()

solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.solve(xs_init, us_init)
xs = solver.xs
us = solver.us
qs = [x[: robot.model.nq] for x in xs]
