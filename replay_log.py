"""Replay a log inside of a visualizer."""
import numpy as np
import quadruped_reactive_walking as qrw

from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer
from quadruped_reactive_walking.wb_mpc.task_spec import TaskSpecBase

import meshcat
import pprint
import example_robot_data as erd

params = qrw.Params.create_from_file()

TMPDIR = Path.home() / ".tmp"
LOGDIR = TMPDIR / "logs"
ALLLOGS = list(LOGDIR.glob("*"))
pprint.pprint(ALLLOGS)

robot = erd.load("solo12")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=Path)
    return parser.parse_args()


args = parse_args()

logfile = open(args.logfile / "data.npz", "rb")
data = np.load(logfile)
KEYS = list(data.keys())

WBC_RATIO = params.mpc_wbc_ratio
ocp_xs = data["ocp_xs"]
ocp_us = data["ocp_us"]

# input x0 to each MPC cycle
dt = params.dt_mpc

psn = data["mocapPosition"]
orn = data["mocapOrientationQuat"]
joints_ = data["q_mes"]
q_rcn = np.concatenate([psn, orn, joints_], axis=1)
q_rcn = q_rcn[::WBC_RATIO]


import hppfcl
import pinocchio as pin

plane = hppfcl.Plane(np.array([0, 0, 1]), 0.)
geobj = pin.GeometryObject("plane", 0, pin.SE3.Identity(), plane)
geobj.meshColor[:] = 0.6, 0.1, 0.1, 0.8
geobj.meshScale *= 1.5

vmodel = robot.visual_model
vmodel.addGeometryObject(geobj)

rmodel = robot.model
vizer = MeshcatVisualizer(rmodel, robot.collision_model, vmodel)
vizer.initViewer(loadModel=True)
vizer.setBackgroundColor()
vizer.display(q_rcn[0])
