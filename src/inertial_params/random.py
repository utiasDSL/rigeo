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


def random_points_on_hypersphere(shape=1, dim=2):
    """Sample random uniform-distributed points on the ``dim``-sphere.

    See https://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    assert dim >= 1
    if np.isscalar(shape):
        shape = (shape,)
    full_shape = tuple(shape) + (dim + 1,)
    X = np.random.normal(size=full_shape)

    # make dimension compatible with X
    r = np.expand_dims(np.linalg.norm(X, axis=-1), axis=X.ndim - 1)

    points = X / r

    # squeeze out extra dimension if shape = 1
    if shape == (1,):
        return np.squeeze(points)
    return points


def random_points_in_ball(shape=1, dim=3):
    """Sample random uniform-distributed points in the ``dim``-ball.

    See https://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    assert dim >= 1

    s = random_points_on_hypersphere(shape=shape, dim=dim - 1)
    c = np.expand_dims(np.random.random(shape), axis=s.ndim - 1)
    points = c ** (1.0 / dim) * s

    # squeeze out extra dimension if shape = 1
    if shape == 1:
        return np.squeeze(points)
    return points
