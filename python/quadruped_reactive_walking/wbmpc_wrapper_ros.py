import numpy as np

from .wb_mpc import get_ocp_from_str
from .wb_mpc.ocp_abstract import OCPAbstract
from .wb_mpc.problem_data import TaskSpec

from typing import Type

from .wbmpc_wrapper_abstract import MPCWrapperAbstract, Result

from threading import Lock

import rospy
from ros_qrw_wbmpc.srv import MPCInit
from std_msgs.msg import MultiArrayDimension, Float64MultiArray, String
import numpy as np

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

        base_refs__multiarray = Float64MultiArray()
        base_refs__multiarray.layout.dim = [MultiArrayDimension('dim%d' % i, base_refs.shape[i], base_refs.shape[i] * base_refs.dtype.itemsize) for i in range(base_refs.ndim)];
        base_refs__multiarray.data = base_refs.reshape([1, -1])[0].tolist()

        footsteps__multiarray = Float64MultiArray()
        footsteps__multiarray.layout.dim = [MultiArrayDimension('dim%d' % i, footsteps.shape[i], footsteps.shape[i] * footsteps.dtype.itemsize) for i in range(footsteps.ndim)];
        footsteps__multiarray.data = footsteps.reshape([1, -1])[0].tolist()

        init_solver_srv = rospy.ServiceProxy("qrw_wbmpc/init", MPCInit)
        self.solver_id = init_solver_srv(String(solver_cls.get_type_str()), String(params.raw_str), base_refs__multiarray, footsteps__multiarray)

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
        pass # Do nothing since it is single threaded


import rospy
from quadruped_reactive_walking import Params
from ros_qrw_wbmpc.srv import MPCInit #, MPCSolve

class ROSMPCWrapperServer:
    def __init__(self):
        self._init_service = rospy.Service('qrw_wbmpc/init', MPCInit, self._trigger_init)

    def _trigger_init(self, msg):
        self.params = Params.create_from_str(msg.params.data)
        self.pd = TaskSpec(self.params)
        self.T = self.params.T
        self.nu = self.pd.nu
        self.nx = self.pd.nx
        self.ndx = self.pd.ndx
        self.solver_cls = get_ocp_from_str(msg.solver_type)

        self.ocp = self.solver_cls(self.params, msg.footsteps, msg.base_refs)

        self.last_available_result: Result = Result(self.params)
        self.new_result = False

        # self._solve_service = rospy.Service('qrw_wbmpc/solve', MPCSolve, self._trigger_solve)

    def _trigger_solve(self, msg):
        pass


if __name__ == "__main__":
    rospy.init_node("qrw_wbmpc")
    server = ROSMPCWrapperServer()
    rospy.spin()
