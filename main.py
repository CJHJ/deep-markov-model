import torch
import numpy as np

import utils.utils as utils


def main():
    data = utils.load_to_numpy('./data/lorenz_96.npy')
    print(data[:10])


if __name__ == "__main__":
    main()
