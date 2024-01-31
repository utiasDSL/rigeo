from pathlib import Path
import numpy as np
import cvxpy as cp

from mobile_manipulation_central import ros_utils


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


def get_urdf_path():
    """Obtain the path to the extra URDFs packaged with pyb_utils.

    Returns
    -------
    : pathlib.Path
        The path to the directory containing extra pyb_utils URDFs.
    """
    return Path(ros_utils.package_file_path("inertial_params", "urdf"))


def schur(A, b, c):
    """Construct the matrix [[A, b], [b.T, c]].

    Parameters can be either cvxpy variables are numpy arrays.
    """
    # cvxpy variables
    if np.any([isinstance(x, cp.Expression) for x in (A, b, c)]):
        b = cp.reshape(b, (b.shape[0], 1))
        c = cp.reshape(c, (1, 1))
        return cp.bmat([[A, b], [b.T, c]])

    # otherwise assume numpy
    b = np.reshape(b, (b.shape[0], 1))
    c = np.reshape(c, (1, 1))
    return np.block([[A, b], [b.T, c]])


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
