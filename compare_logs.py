import numpy as np
import glob
from pathlib import Path
import argparse

TMPDIR = Path.home() / ".tmp"
print(TMPDIR)

ALLLOGS = list(TMPDIR.glob("logs/2023_*"))
print(ALLLOGS)

parser = argparse.ArgumentParser()
parser.add_argument("log1", type=Path)
parser.add_argument("log2", type=Path)
args = parser.parse_args()

print(args)

assert args.log1 in ALLLOGS
assert args.log2 in ALLLOGS

file1 = open(args.log1 / "data.npz", "rb")
file2 = open(args.log2 / "data.npz", "rb")

data1 = np.load(file1)
data2 = np.load(file2)

KEYS = list(data1.keys())


assert "ocp_xs" in KEYS
assert "ocp_us" in KEYS


import matplotlib.pyplot as plt
import example_robot_data as erd
import crocoddyl
from matplotlib import colors
import pinocchio as pin

robot = erd.load("solo12")
rmodel = robot.model
state = crocoddyl.StateMultibody(rmodel)

plt.rcParams["lines.linewidth"] = 1.

xs1 = data1['ocp_xs']
us1 = data1['ocp_us']

NITER = us1.shape[0]
NSTEPS = us1.shape[1]

xs2 = data2['ocp_xs']
us2 = data2['ocp_us']


Xerr = np.zeros((NITER, NSTEPS + 1, state.ndx))
for i in range(NITER):
    for j in range(NSTEPS + 1):
        Xerr[i, j, :] = state.diff(xs1[i, j], xs2[i, j])
Xerr_over_mpc = np.linalg.norm(Xerr, axis=(1, 2))
print("Xerr:", Xerr_over_mpc)

plt.plot(Xerr_over_mpc)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.yscale("log")
plt.title("Comparison between\n{:s} and\n{:s}".format(str(args.log1), str(args.log2)))

# Error over time & mpc iteration
Xerr_over_time = np.linalg.norm(Xerr, axis=2)
norm = colors.LogNorm()

aspect = NSTEPS / NITER
plt.figure()
im = plt.imshow(Xerr_over_time, cmap="viridis", norm=norm, aspect=aspect)
plt.colorbar()
plt.ylabel("MPC iteration")
plt.xlabel("Time $t$")
plt.title("state error")

Uerr = us2 - us1
Uerr_over_time = np.linalg.norm(Uerr, axis=2)
norm = colors.LogNorm()

plt.figure()
im = plt.imshow(Uerr_over_time, cmap="viridis", norm=norm, aspect=aspect)
plt.colorbar()
plt.ylabel("MPC iteration")
plt.xlabel("Time $t$")
plt.title("ctrl error")

# Error over first MPC iterations
k_mpc = np.arange(0, 90, 15)
Xerr_over_1st_loop = Xerr[k_mpc, :, :]
Xerr_over_1st_loop = np.linalg.norm(Xerr_over_1st_loop, axis=2)
plt.figure()
lines = plt.plot(Xerr_over_1st_loop.T)
plt.legend(lines, k_mpc.tolist())
plt.xlabel("Time $t$")
plt.ylabel("Err")
plt.yscale("log")
plt.title("Error over 1st MPC iteration")


Uerr_over_1st_loop = Uerr[k_mpc, :, :]
Uerr_over_1st_loop = np.linalg.norm(Uerr_over_1st_loop, axis=2)
plt.figure()
plt.plot(Uerr_over_1st_loop.T)
plt.xlabel("Time $t$")
plt.yscale("log")
plt.title("Ctrl error over first iters")


plt.show()