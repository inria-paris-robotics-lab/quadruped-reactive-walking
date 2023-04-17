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
        self.solve_solver_srv(MPCSolveRequest())


class TestServer:
    def __init__(self):
        self._solve_service = rospy.Service(
            "qrw_wbmpc/test", MPCSolve, self._trigger_solve
        )

    def _trigger_solve(self, msg):
        return MPCSolveResponse(run_success=False)


if __name__ == "__main__":
    rospy.init_node("qrw_wbmpc")
    server = TestServer()
    rospy.spin()
