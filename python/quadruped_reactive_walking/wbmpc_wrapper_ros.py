from quadruped_reactive_walking import Params

from .wb_mpc import get_ocp_from_str
from .wb_mpc.ocp_abstract import OCPAbstract
from .wb_mpc.task_spec import TaskSpec

from .wbmpc_wrapper_abstract import MPCWrapperAbstract, Result

from typing import Type
from threading import Lock
import numpy as np

import rospy
from ros_qrw_wbmpc.srv import MPCInit, MPCInitResponse, MPCSolve, MPCSolveResponse
from .tools.ros_tools import (
    numpy_to_multiarray_float64,
    multiarray_to_numpy_float64,
    AsyncServiceProxy,
)


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
        self.new_result: bool = False
        self.last_available_result: Result = Result(params)

        base_refs_multiarray = numpy_to_multiarray_float64(base_refs)
        footsteps_multiarray = numpy_to_multiarray_float64(footsteps)

        init_solver_srv = rospy.ServiceProxy("qrw_wbmpc/init", MPCInit)
        success = init_solver_srv(
            solver_type=solver_cls.get_type_str(),
            params=params.raw_str,
            footsteps=footsteps_multiarray,
            base_refs=base_refs_multiarray,
        )
        assert success, "Error while initializing mpc on server"

        self.solve_solver_srv = AsyncServiceProxy(
            "qrw_wbmpc/solve", MPCSolve, callback=self._result_cb
        )

    def solve(self, k, x0, footstep, base_ref, xs=None, us=None):
        self.solve_solver_srv(
            k=k,
            x0=numpy_to_multiarray_float64(x0),
            footstep=numpy_to_multiarray_float64(footstep),
            base_ref=numpy_to_multiarray_float64(base_ref),
            xs=numpy_to_multiarray_float64(np.array(xs if xs is not None else [])),
            us=numpy_to_multiarray_float64(np.array(us if us is not None else [])),
        )

    def _result_cb(self, fut):
        msg = fut.result()
        with self._result_lock:
            self.new_result = True
            self.last_available_result.P = multiarray_to_numpy_float64(msg.P)
            self.last_available_result.D = multiarray_to_numpy_float64(msg.D)
            self.last_available_result.FF = multiarray_to_numpy_float64(msg.FF)
            self.last_available_result.q_des = multiarray_to_numpy_float64(msg.q_des)
            self.last_available_result.v_des = multiarray_to_numpy_float64(msg.v_des)
            self.last_available_result.FF = multiarray_to_numpy_float64(msg.FF)

    def get_latest_result(self):
        """
        Return the desired contact forces that have been computed by the last iteration
        of the MPC.
        If a new result is available, return the new result.
        Otherwise return the old result again.
        """
        with self._result_lock:
            self.last_available_result.new_result = self.new_result
            self.new_result = False

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
        self._solve_service = rospy.Service(
            "qrw_wbmpc/solve", MPCSolve, self._trigger_solve
        )

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

        footsteps = multiarray_to_numpy_float64(msg.footsteps)
        base_refs = multiarray_to_numpy_float64(msg.base_refs)

        self.ocp = self.solver_cls(self.params, footsteps, base_refs)

        self.last_available_result: Result = Result(self.params)

        return MPCInitResponse(True)

    def _trigger_solve(self, msg):
        self.ocp.make_ocp(
            msg.k,
            multiarray_to_numpy_float64(msg.x0),
            multiarray_to_numpy_float64(msg.footstep),
            multiarray_to_numpy_float64(msg.base_ref),
        )

        xs = multiarray_to_numpy_float64(msg.xs)
        us = multiarray_to_numpy_float64(msg.us)

        self.ocp.solve(msg.k, xs, us)

        result = self.ocp.get_results()

        return MPCSolveResponse(
            P=numpy_to_multiarray_float64(result.P),
            D=numpy_to_multiarray_float64(result.D),
            FF=numpy_to_multiarray_float64(result.FF),
            q_des=numpy_to_multiarray_float64(result.q_des),
            v_des=numpy_to_multiarray_float64(result.v_des),
            tau_ff=numpy_to_multiarray_float64(result.tau_ff),
        )


if __name__ == "__main__":
    rospy.init_node("qrw_wbmpc")
    server = ROSMPCWrapperServer()
    rospy.spin()
