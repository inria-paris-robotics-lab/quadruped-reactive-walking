from .wb_mpc import get_ocp_from_str
from .wb_mpc.ocp_abstract import OCPAbstract
from .wb_mpc.problem_data import TaskSpec

from typing import Type

from .wbmpc_wrapper_abstract import MPCWrapperAbstract, Result

from threading import Lock

from std_msgs.msg import MultiArrayDimension, Float64MultiArray
import numpy as np

import rospy
from ros_qrw_wbmpc.srv import MPCInit, MPCInitResponse  # , MPCSolve
from quadruped_reactive_walking import Params


def array_np_to_ros_float64(np_array):
    multiarray = Float64MultiArray()
    multiarray.layout.dim = [
        MultiArrayDimension(
            "dim%d" % i, np_array.shape[i], np_array.shape[i] * np_array.dtype.itemsize
        )
        for i in range(np_array.ndim)
    ]
    multiarray.data = np_array.reshape([1, -1])[0].tolist()


def array_ros_to_np_float64(ros_array):
    dims = [d.size for d in ros_array.layout.dim]
    if dims == []:
        return np.array([])
    return np.array(ros_array.data).reshape(dims)


class ROSMPCWrapperClient(MPCWrapperAbstract):
    """
    Wrapper to run both types of MPC (OQSP or Crocoddyl) in a synchronous manner in the main thread.
    """

    def __init__(
        self, params, footsteps, base_refs, solver_cls: Type[OCPAbstract], **kwargs
    ):
        footsteps = np.array(footsteps)
        base_refs = np.array(base_refs)

        self._result_lock = Lock()

        base_refs__multiarray = array_np_to_ros_float64(base_refs)
        footsteps__multiarray = array_np_to_ros_float64(footsteps)

        init_solver_srv = rospy.ServiceProxy("qrw_wbmpc/init", MPCInit)
        self.solver_id = init_solver_srv(
            solver_cls.get_type_str(),
            params.raw_str,
            base_refs__multiarray,
            footsteps__multiarray,
        )

        # rospy.SubscribeListener("Result", ...)
        # rospy.Publisher("Solve", ...)

    def solve(self, k, x0, footstep, base_ref, xs=None, us=None):
        print("======= Solve =======")
        print("k", k, "\n")
        print("x0", x0, "\n")
        print("footstep", footstep, "\n")
        print("base_ref", base_ref, "\n")
        print("xs", us, "\n")
        print("us", xs, "\n")
        # pub.publish
        pass

    def _result_cb(self, msg):
        with self._result_lock:
            self.new_result = True
            self.last_available_result = msg

    def get_latest_result(self):
        """
        Return the desired contact forces that have been computed by the last iteration
        of the MPC.
        If a new result is available, return the new result.
        Otherwise return the old result again.
        """
        with self._result_lock:
            if self.new_result:
                self.last_available_result.new_result = True
                self.new_result = False
            else:
                self.last_available_result.new_result = False

        return self.last_available_result

    def stop_parallel_loop(self):
        # rospy.pub("Kill")
        pass  # Do nothing since it is single threaded


class ROSMPCWrapperServer:
    def __init__(self):
        self.is_init = False
        self._init_service = rospy.Service(
            "qrw_wbmpc/init", MPCInit, self._trigger_init
        )
        # self._solve_service = rospy.Service(
        #     "qrw_wbmpc/solve", MPCSolve, self._trigger_solve
        # )

    def _trigger_init(self, msg):
        if self.is_init:
            return MPCInitResponse(False)

        self.is_init = True
        self.params = Params.create_from_str(msg.params)
        self.pd = TaskSpec(self.params)
        self.T = self.params.N_gait
        self.nu = self.pd.nu
        self.nx = self.pd.nx
        self.ndx = self.pd.ndx
        self.solver_cls = get_ocp_from_str(msg.solver_type)

        footsteps = array_ros_to_np_float64(msg.footsteps)
        base_refs = array_ros_to_np_float64(msg.base_refs)

        self.ocp = self.solver_cls(self.params, footsteps, base_refs)

        self.last_available_result: Result = Result(self.params)
        self.new_result = False

        return MPCInitResponse(True)

    def _trigger_solve(self, msg):
        pass


if __name__ == "__main__":
    rospy.init_node("qrw_wbmpc")
    server = ROSMPCWrapperServer()
    rospy.spin()
