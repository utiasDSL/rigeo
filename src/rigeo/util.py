from pathlib import Path
import numpy as np


def vech(J):
    """Half-vectorize the inertia matrix"""
    return np.array([J[0, 0], J[0, 1], J[0, 2], J[1, 1], J[1, 2], J[2, 2]])


def skew3(v):
    """Form a skew-symmetric matrix out of 3-dimensional vector v."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def skew6(V):
    """6D cross product matrix"""
    v, ω = V[:3], V[3:]
    Sv = skew3(v)
    Sω = skew3(ω)
    return np.block([[Sω, np.zeros((3, 3))], [Sv, Sω]])


def lift3(x):
    """Lift a 3-vector x such that A @ x = lift(x) @ vech(A) for symmetric A."""
    # fmt: off
    return np.array([
        [x[0], x[1], x[2], 0, 0, 0],
        [0, x[0], 0, x[1], x[2], 0],
        [0, 0, x[0], 0, x[1], x[2]]
    ])
    # fmt: on


def lift6(x):
    """Lift a 6-vector V such that A @ V = lift(V) @ vech(A) for symmetric A."""
    a = x[:3]
    b = x[3:]
    # fmt: off
    return np.block([
        [a[:, None], skew3(b), np.zeros((3, 6))],
        [np.zeros((3, 1)), -skew3(a), lift3(b)]])
    # fmt: on


def compute_evaluation_times(duration, step=0.1):
    """Compute times spaced at a fixed step across an interval.

    Times start at zero.

    Parameters
    ----------
    duration : float, positive
        Duration of the interval.
    step : float, positive
        Duration of each step.

    Returns
    -------
    : tuple
        A tuple ``(n, times)`` representing the number of times and the time
        values themselves.
    """
    assert duration > 0
    assert step > 0

    n = int(np.ceil(duration / step))
    times = step * np.arange(n)
    return n, times


def validation_rmse(Ys, ws, θ):
    """Compute root mean square wrench error on a validation set."""
    error = Ys @ θ - ws
    square = np.sum(error**2, axis=1)
    mean = np.mean(square)
    root = np.sqrt(mean)
    return root


def clean_transform(rotation, translation, dim):
    if rotation is None:
        rotation = np.eye(dim)
    else:
        rotation = np.array(rotation)

    if translation is None:
        translation = np.zeros(dim)
    else:
        translation = np.array(translation)

    return rotation, translation