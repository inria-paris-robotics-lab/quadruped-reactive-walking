import hppfcl
import numpy as np
import pinocchio as pin

from pinocchio import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer


class MeshcatViewer:
    def __init__(self, robot: RobotWrapper):
        self.rmodel = robot.model
        self.rdata = robot.data
        plane_normal = np.array([0.0, 0.0, 1.0])
        plane = hppfcl.Plane(plane_normal, 0.0)
        plane_obj = pin.GeometryObject("ground", 0, pin.SE3.Identity(), plane)
        plane_obj.meshColor[:] = 0.7, 1.0, 0.65, 1.0
        robot.visual_model.addGeometryObject(plane_obj)
        robot.collision_model.addGeometryObject(plane_obj)

        self.viz = MeshcatVisualizer(
            self.rmodel,
            collision_model=robot.collision_model,
            visual_model=robot.visual_model,
            data=robot.data,
        )
        self.viz.initViewer(loadModel=True, open=True)
        self.viz.setBackgroundColor()

    def update(self, controller, device):
        q = np.zeros(19)
        q[7:] = device.joints.positions
        q[:7] = pin.SE3ToXYZQUAT(device.oMb)
        q[2] = device.baseState[2]
        self.viz.display(q)
