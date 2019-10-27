from scipy.integrate import odeint
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def f(x, t, N, F):
    # Initialize derivatives
    d = np.zeros(N)

    # Edge cases
    d[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0]
    d[1] = (x[2] - x[N - 1]) * x[0] - x[1]
    d[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1]

    # General case
    for i in range(2, N - 1):
        d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]

    # Add forcing term
    d = d + F

    return d


def generate(N=20, F=8.0, tf=30.0, dt=0.01):
    # Initialize state and time list
    x0 = F * np.zeros(N)
    x0[N - 2] += 0.01
    t = np.arange(0.0, tf, dt)

    x = odeint(f, x0, t, args=(N, F))

    return x


def plot(data, start=0, end=2000):
    ax = sns.heatmap(data[start:end:, :])
    ax.invert_yaxis()
    plt.show()


def save_to_numpy(data, file_path):
    np.save(file_path, data)


def main():
    data = generate(N=40, F=8.0)
    save_to_numpy(data, './data/lorenz_96.npy')


if __name__ == "__main__":
    main()
