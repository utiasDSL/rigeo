import numpy as np
from scipy.linalg import sqrtm


def random_psd_matrix(n):
    A = 2 * np.random.random((n, n)) - 1
    return sqrtm(A.T @ A)


def random_weight_vectors(shape):
    """Generate a set of random vectors with non-negative entries that sum to one.

    Vectors sum to one along the last axis of ``shape``. Entries are uniformly
    distributed.

    Parameters
    ----------
    shape : int or tuple
        Shape of the weight vector(s).

    Returns
    -------
    : np.ndarray
        A set of weight vectors.
    """
    w = np.random.random(shape)
    s = np.expand_dims(np.sum(w, axis=-1), axis=w.ndim - 1)
    return w / s
