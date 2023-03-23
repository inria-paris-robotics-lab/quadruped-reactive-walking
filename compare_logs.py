import numpy as np
from pathlib import Path
import quadruped_reactive_walking as qrw
import argparse
import crocoddyl
import example_robot_data as erd

robot = erd.load("solo12")
rmodel = robot.model
state = crocoddyl.StateMultibody(rmodel)

params = qrw.Params.create_from_file()
wbc_ratio = params.mpc_wbc_ratio
print(wbc_ratio)

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
from matplotlib import colors

plt.rcParams["lines.linewidth"] = 1.

xs1 = data1['ocp_xs']
us1 = data1['ocp_us']

NITER = us1.shape[0]
NSTEPS = us1.shape[1]

xs2 = data2['ocp_xs']
us2 = data2['ocp_us']

NJOINTS = 12
fbs1 = data1['ocp_K']
fbs2 = data2['ocp_K']

assert fbs1.shape[1] == NJOINTS
assert fbs1.shape[2] == state.ndx

# Plotting
def infNorm(x, **kwargs):
    return np.linalg.norm(x, ord=np.inf, **kwargs)

Xerr = np.zeros((NITER, NSTEPS + 1, state.ndx))
for i in range(NITER):
    for j in range(NSTEPS + 1):
        Xerr[i, j, :] = state.diff(xs1[i, j], xs2[i, j])
Xerr_over_mpc = infNorm(Xerr[::wbc_ratio], axis=(1, 2))

K_err = fbs2 - fbs1
K_err = K_err[::wbc_ratio]
K_err_over_mpc = infNorm(K_err, axis=(1, 2))

plt.figure()
plt.plot(Xerr_over_mpc, label="State err")
plt.plot(K_err_over_mpc, label="Feedback err")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.yscale("log")
plt.legend()
plt.title("Comparison between\n{:s} and\n{:s}".format(str(args.log1), str(args.log2)))

# Error over time & mpc iteration
Xerr_over_time = infNorm(Xerr[::wbc_ratio], axis=2)
norm = colors.LogNorm()

aspect = NSTEPS / (NITER // wbc_ratio)
plt.figure()
im = plt.imshow(Xerr_over_time, cmap="viridis", norm=norm, aspect=aspect)
plt.colorbar()
plt.ylabel("MPC iteration")
plt.xlabel("Time $t$")
plt.title("state error")

Uerr = us2 - us1
Uerr_over_time = infNorm(Uerr[::wbc_ratio], axis=2)
norm = colors.LogNorm()

plt.figure()
im = plt.imshow(Uerr_over_time, cmap="viridis", norm=norm, aspect=aspect)
plt.colorbar()
plt.ylabel("MPC iteration")
plt.xlabel("Time $t$")
plt.title("ctrl error")

# Error over first MPC iterations
k_mpc = np.arange(0, 90, wbc_ratio)
Xerr_over_1st_loop = Xerr[k_mpc, :, :]
Xerr_over_1st_loop = infNorm(Xerr_over_1st_loop, axis=2)
plt.figure()
lines = plt.plot(Xerr_over_1st_loop.T)
plt.legend(lines, k_mpc.tolist(), title="iteration")
plt.xlabel("Time $t$")
plt.ylabel("Err")
plt.yscale("log")
plt.title("State error over first MPC iterations")


Uerr_over_1st_loop = Uerr[k_mpc, :, :]
Uerr_over_1st_loop = np.linalg.norm(Uerr_over_1st_loop, axis=2)
plt.figure()
lines = plt.plot(Uerr_over_1st_loop.T)
plt.legend(lines, k_mpc.tolist(), title='iteration')
plt.xlabel("Time $t$")
plt.yscale("log")
plt.title("Ctrl error over first iters")


FIGDIR = Path("figs")
FIGDIR.mkdir(exist_ok=True)
figs = [plt.figure(n) for n in plt.get_fignums()]
labs = ["errors_per_iter", "imshow_state", "imshow_control", "state_err_traj_mpc", "control_err_traj_mpc"]
for fig, l in zip(figs, labs):
    fpath = (FIGDIR / l).with_suffix(".png")
    print(fpath)
    fig.savefig(fpath)
