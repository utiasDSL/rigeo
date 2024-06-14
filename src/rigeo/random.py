"""Generate random values."""
import numpy as np


def random_psd_matrix(n):
    """Generate a random symmetric positive semidefinite matrix.

    Parameters
    ----------
    n : int
        Dimension of the matrix.

    Returns
    -------
    : np.ndarray, shape (n, n)
        A positive semidefinite matrix.
    """
    # TODO: could look into doi.org/10.1109/TSP.2012.2186447 for
    # uniform-sampling of trace-constrained PSD matrices
    A = 2 * np.random.random((n, n)) - 1
    return A.T @ A


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


def rejection_sample(actual_shapes, bounding_shape, sample_shape, max_tries=10000):
    if np.isscalar(sample_shape):
        sample_shape = (sample_shape,)
    sample_shape = tuple(sample_shape)

    n = np.prod(sample_shape)
    full = np.zeros(n, dtype=bool)
    points = np.zeros((n, 3))
    m = 0
    tries = 0
    while m < n:
        # generate as many points as we still need
        candidates = bounding_shape.random_points(n - m)

        # check if they are contained in the actual shape
        # TODO this is wrong
        # TODO this needs to be tested
        # c = np.any([s.contains(candidates) for s in actual_shapes])
        c = np.sum([s.contains(candidates) for s in actual_shapes], axis=0).astype(bool)

        # short-circuit if no points are contained
        if not c.any():
            continue

        # use the points that are contained, storing them and marking them
        # full
        new_points = candidates[c]
        n_new = new_points.shape[0]
        points[m : m + n_new] = new_points

        # update count of remaining points to generate
        # m = n - np.sum(full)
        m += n_new

        # eventually error out if this is taking too long
        tries += 1
        if tries >= max_tries:
            raise ValueError("Failed to generate enough points by rejection sampling.")

    # back to original shape
    if sample_shape == (1,):
        return np.squeeze(points)
    return points.reshape(sample_shape + (3,))
