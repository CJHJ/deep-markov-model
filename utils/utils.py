import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def save_as_numpy(data, file_path):
    np.save(file_path, data)


def load_to_numpy(file_path):
    data = np.load(file_path)

    return data


def plot(data, start=0, end=2000):
    ax = sns.heatmap(data[start:end:, :])
    ax.invert_yaxis()
    plt.show()
