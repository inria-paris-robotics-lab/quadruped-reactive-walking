import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


class Interpolator:
    def __init__(self, params, x0):
        self.dt = params.dt_mpc
        self.type = params.interpolation_type

        if self.type == 3:
            self.ts = np.repeat(np.linspace(0, 2 * self.dt, 3), 2)

        self.update(x0, x0)

    def update(self, x0, x1, x2=None):
        self.q0 = x0[7:19]
        self.q1 = x1[7:19]
        self.v0 = x0[19 + 6 :]
        self.v1 = x1[19 + 6 :]
        if self.type == 0:  # Linear
            self.alpha = 0.0
            self.beta = self.v1
            self.gamma = self.q0
        elif self.type == 1:  # Quadratic fixed velocity
            self.alpha = 2 * (self.q1 - self.q0 - self.v0 * self.dt) / self.dt**2
            self.beta = self.v0
            self.gamma = self.q0
        elif self.type == 2:  # Quadratic time variable
            for i in range(3):
                q0 = self.q0[i]
                v0 = self.v0[i]
                q1 = self.q1[i]
                v1 = self.v1[i]
                if (q1 == q0) or (v1 == -v0):
                    self.alpha[i] = 0.0
                    self.beta[i] = 0.0
                    self.gamma[i] = q1
                    self.delta = 1.0
                else:
                    self.alpha[i] = (v1**2 - v0**2) / (2 * (q1 - q0))
                    self.beta[i] = v0
                    self.gamma[i] = q0
                    self.delta = 2 * (q1 - q0) / (v1 + v0) / self.dt
        elif self.type == 3:  # Spline interpolation
            from scipy.interpolate import KroghInterpolator

            if x2 is not None:
                self.q2 = x2[7:19]
                self.v2 = x2[19 + 6 :]
                self.y = [self.q0, self.v0, self.q1, self.v1, self.q2, self.v2]
                self.krog = KroghInterpolator(self.ts, np.array(self.y))
            else:
                self.y = [self.q0, self.v0, self.q1, self.v1]
                self.krog = KroghInterpolator(self.ts[:4], np.array(self.y))

    def interpolate(self, t):
        if self.type == 3:
            q = self.krog(t)
            v = self.krog.derivative(t)
            return q, v

        if self.type == 2:
            t *= self.delta
        q = 1 / 2 * self.alpha * t**2 + self.beta * t + self.gamma
        v = self.v1 if self.type == 1 else self.alpha * t + self.beta

        return q, v


def plot(interp: Interpolator, n):
    import matplotlib.pyplot as plt

    ts = np.linspace(0.0, 2 * interp.dt, 2 * n + 1)
    plt.style.use("seaborn")
    for i in range(3):
        plt.subplot(3, 2, (i * 2) + 1)
        plt.title("Position interpolation")
        plt.plot(ts, [interp.interpolate(t)[0][i] for t in ts])
        plt.scatter(y=interp.q0[i], x=0.0, color="violet", marker="+")
        plt.scatter(y=interp.q1[i], x=interp.dt, color="violet", marker="+")
        if interp.type == 3 and interp.q2 is not None:
            plt.scatter(y=interp.q2[i], x=2 * interp.dt, color="violet", marker="+")

        plt.subplot(3, 2, (i * 2) + 2)
        plt.title("Velocity interpolation")
        plt.plot(ts, [interp.interpolate(t)[1][i] for t in ts])
        plt.scatter(y=interp.v0[i], x=0.0, color="violet", marker="+")
        plt.scatter(y=interp.v1[i], x=interp.dt, color="violet", marker="+")
        if interp.type == 3 and interp.v2 is not None:
            plt.scatter(y=interp.v2[i], x=2 * interp.dt, color="violet", marker="+")

    plt.show()
