import ctypes
from ctypes import Structure
from enum import Enum
from multiprocessing import Process, Value, Array
from time import time

import numpy as np

from .WB_MPC.CrocoddylOCP import OCP

import quadruped_reactive_walking as qrw


class DataInCtype(Structure):
    """
    Ctype data structure for the shared memory between processes.
    """

    params = qrw.Params()
    # TODO add the data exchanged with the OCP
    _fields_ = [
        ("k", ctypes.c_int64),
        ("x0", ctypes.c_double * 6),
        ("guess", ctypes.c_double * 12),
    ]


class MPC_Wrapper:
    """
    Wrapper to run both types of MPC (OQSP or Crocoddyl) with the possibility to run OSQP in
    a parallel process
    """

    def __init__(self, pd, target, params):
        self.initialized = False
        self.params = params
        self.pd = pd
        self.target = target

        n_nodes = 0

        self.multiprocessing = params.enable_multiprocessing

        if self.multiprocessing:
            self.newData = Value("b", False)
            self.newResult = Value("b", False)
            self.running = Value("b", True)
            self.dataIn = Value(DataInCtype)
            # TODO: update output size
            self.dataOut = Array("d", [0] * 24 * n_nodes)
            self.cost = Value("d", 0.0)
        else:
            self.ocp = OCP(pd, target)

        # TODO initialize first result
        self.last_available_result = np.zeros((24, n_nodes))
        self.last_cost = 0.0

    def solve(self, k, x0, guess=None):
        """
        Call either the asynchronous MPC or the synchronous MPC depending on the value
        of multiprocessing during the creation of the wrapper

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
        """

        if self.multiprocessing:
            self.run_MPC_asynchronous(k, x0, guess)
        else:
            self.run_MPC_synchronous(x0, guess)

    def get_latest_result(self):
        """
        Return the desired contact forces that have been computed by the last iteration
        of the MPC.
        If a new result is available, return the new result. 
        Otherwise return the old result again.
        """
        if self.initialized:
            if self.multiprocessing and self.newResult.value:
                self.newResult.value = False
                self.last_available_result = self.convert_dataOut()
                self.last_cost = self.cost.value
        else:
            self.initialized = True
        return self.last_available_result, self.last_cost

    def run_MPC_synchronous(self, x0, guess):
        """
        Run the MPC (synchronous version)
        """
        self.ocp.solve(x0, guess)
        self.last_available_result = self.ocp.get_results()
        #self.last_cost = self.ocp.retrieve_cost()

    def run_MPC_asynchronous(self, k, x0, guess):
        """
        Run the MPC (asynchronous version)
        """
        if k == 0:
            p = Process(target=self.MPC_asynchronous)
            p.start()

        self.add_new_data(k, x0, guess)

    def MPC_asynchronous(self):
        """
        Parallel process with an infinite loop that run the asynchronous MPC
        """
        while self.running.value:
            if self.newData.value:
                self.newData.value = False

                k, x0, guess = self.decompress_dataIn(self.dataIn)

                if k == 0:
                    loop_ocp = OCP(self.pd, self.target)

                loop_ocp.solve(x0, guess)
                self.dataOut[:] = loop_ocp.get_latest_result().ravel(order="F")
                self.cost.value = loop_ocp.retrieve_cost()
                self.newResult.value = True

    def add_new_data(self, k, x0):
        """
        Compress data in a C-type structure that belongs to the shared memory to send 
        data from the main control loop to the asynchronous MPC and notify the process
        that there is a new data
        """
        with self.dataIn.get_lock():
            self.dataIn.k = k
            # np.frombuffer(self.dataIn.xref).reshape((12, self.n_steps + 1))[
            #         :, :
            #     ] = xref
        self.newData.value = True

    def decompress_dataIn(self, dataIn):
        """
        Decompress data from a C-type structure that belongs to the shared memory to
        retrieve data from the main control loop in the asynchronous MPC

        Args:
            dataIn (Array): shared C-type structure that contains the input data
        """

        with dataIn.get_lock():
            k = self.dataIn.k
            xref = np.frombuffer(self.dataIn.xref).reshape((12, self.n_steps + 1))

            return k, xref

    def convert_dataOut(self):
        """
        Return the result of the asynchronous MPC (desired contact forces) that is
        stored in the shared memory
        """

        return np.array(self.dataOut[:]).reshape((24, -1), order="F")

    def stop_parallel_loop(self):
        """
        Stop the infinite loop in the parallel process to properly close the simulation
        """

        self.running.value = False

        return 0
