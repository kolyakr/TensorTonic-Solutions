import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """

    np_x = np.array(x)
    exp_x = np.exp(-np_x)

    return 1 / (1 + exp_x)