from quadruped_reactive_walking import Params


class OCPBuilder:
    def __init__(self, params: Params) -> None:
        self.params = params
        self.task = None
        self.problem = None
