import abc
import numpy as np

from .ProblemData import ProblemData
import quadruped_reactive_walking as qrw


class OCPAbstract(abc.ABC):
    def __init__(self, pd: ProblemData, params: qrw.Params):
        self.pd = pd
        self.params = params
