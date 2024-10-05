"""Generate random values."""
import numpy as np


def random_psd_matrix(n, rng=None):
    """Generate a random symmetric positive semidefinite matrix.

    Parameters
    ----------
    n : int
        Dimension of the matrix.
    rng : int or np.random.Generator
        Integer seed or Generator instance to use for generating random
        numbers.

    Returns
    -------
    : np.ndarray, shape (n, n)
        A positive semidefinite matrix.
    """
    # TODO: could look into doi.org/10.1109/TSP.2012.2186447 for
    # uniform-sampling of trace-constrained PSD matrices
    rng = np.random.default_rng(rng)
    A = rng.uniform(low=-1, high=1, size=(n, n))
    return A.T @ A


def random_weight_vectors(shape, rng=None):
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
    rng = np.random.default_rng(rng)
    w = rng.random(shape)
    s = np.expand_dims(np.sum(w, axis=-1), axis=w.ndim - 1)
    return w / s


def random_points_on_hypersphere(shape=1, dim=2, rng=None):
    """Sample random uniform-distributed points on the ``dim``-sphere.

    See https://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    assert dim >= 1
    if np.isscalar(shape):
        shape = (shape,)
    full_shape = tuple(shape) + (dim + 1,)

    rng = np.random.default_rng(rng)
    X = rng.normal(size=full_shape)

    # make dimension compatible with X
    r = np.expand_dims(np.linalg.norm(X, axis=-1), axis=X.ndim - 1)

    points = X / r

    # squeeze out extra dimension if shape = 1
    if shape == (1,):
        return np.squeeze(points)
    return points


def random_points_in_ball(shape=1, dim=3, rng=None):
    """Sample random uniform-distributed points in the ``dim``-ball.

    See https://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    assert dim >= 1

    rng = np.random.default_rng(rng)
    s = random_points_on_hypersphere(shape=shape, dim=dim - 1, rng=rng)
    c = np.expand_dims(rng.random(shape), axis=s.ndim - 1)
    points = c ** (1.0 / dim) * s

    # squeeze out extra dimension if shape = 1
    if shape == 1:
        return np.squeeze(points)
    return points


def rejection_sample(
    actual_shapes, bounding_shape, sample_shape, max_tries=10000, rng=None
):
    if np.isscalar(sample_shape):
        sample_shape = (sample_shape,)
    sample_shape = tuple(sample_shape)

    rng = np.random.default_rng(rng)

    n = np.prod(sample_shape)  # number of points required
    m = 0  # number of points generated so far
    points = np.zeros((n, 3))
    tries = 0
    while m < n:
        # eventually error out if this is taking too long
        if tries >= max_tries:
            raise ValueError(
                "Failed to generate enough points by rejection sampling."
            )
        tries += 1

        # generate as many points as we still need
        candidates = bounding_shape.random_points(n - m, rng=rng)

        # check if they are contained in the actual shape
        # any value >= 1 will be cast to True
        c = np.sum(
            [s.contains(candidates) for s in actual_shapes], axis=0
        ).astype(bool)

        # short-circuit if no points are contained
        if not c.any():
            continue

        # use the points that are contained in at least one of the shapes
        new_points = candidates[c]
        n_new = new_points.shape[0]
        points[m : m + n_new] = new_points

        # update count of remaining points to generate
        m += n_new

    # back to original shape
    if sample_shape == (1,):
        return np.squeeze(points)
    return points.reshape(sample_shape + (3,))
