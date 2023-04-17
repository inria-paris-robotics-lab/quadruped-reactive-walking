import rospy
from quadruped_reactive_walking.srv import (
    MPCSolve,
    MPCSolveRequest,
    MPCSolveResponse,
)


class TestClient:
    """
    Wrapper to run both types of MPC (OQSP or Crocoddyl) on a seperate node/machine using ROS as communication interface.
    """

    def __init__(
        self,
    ):
        self.solve_solver_srv = rospy.ServiceProxy(
            "qrw_wbmpc/test", MPCSolve, persistent=True
        )

    def solve(self):
        return self.solve_solver_srv(MPCSolveRequest())


class TestServer:
    def __init__(self):
        self._solve_service = rospy.Service(
            "qrw_wbmpc/test", MPCSolve, self._trigger_solve
        )

    def _trigger_solve(self, msg):
        return MPCSolveResponse(run_success=True)


def fake_data():
    import numpy as np
    from quadruped_reactive_walking.tools.ros_tools import numpy_to_multiarray_float64

    k = 0
    x0 = np.random.randn(37)
    footstep = np.random.randn(3, 4)
    base_ref = np.array([0.5, 0, 0, 0, 0, 0])
    req = MPCSolveRequest(
        k=k,
        x0=numpy_to_multiarray_float64(x0),
        footstep=numpy_to_multiarray_float64(footstep),
        base_ref=numpy_to_multiarray_float64(base_ref),
    )
    return req


if __name__ == "__main__":
    rospy.init_node("qrw_wbmpc")
    server = TestServer()
    rospy.spin()
