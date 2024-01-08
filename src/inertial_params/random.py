import numpy as np
from scipy.linalg import sqrtm


def random_psd_matrix(shape):
    A = 2 * np.random.random(shape) - 1
    return sqrtm(A.T @ A)
