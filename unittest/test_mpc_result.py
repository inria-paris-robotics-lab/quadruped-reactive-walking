from quadruped_reactive_walking import MPCResult
import numpy as np

ng = 4
res = MPCResult(ng, 4, 2, 4)

us = [np.random.randn(2) for _ in range(ng)]
res.us = us

# should raise ValueError
try:
    res.us = us[:2]
    assert False
except ValueError:
    pass
