"""Miscellaneous utility functions."""
import itertools
import numbers
from pathlib import Path

import numpy as np


def vech(A, k=0):
    """Half-vectorize a matrix.

    This extracts a flattened (vector) representation of the upper triangular
    part of the matrix.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        The matrix to half-vectorize.
    k : int
        Diagonal offset; e.g., passing ``k=1`` vectorizes the *strictly* upper
        triangular values.

    Returns
    -------
    : np.ndarray
        The vector of upper triangular values.
    """
    n, m = A.shape
    assert n == m
    idx = np.triu_indices(n, k=k)
    return A[idx]


def is_triangular_number(x):
    """Check if a number is a triangular number.

    That is, there exists a positive integer ``n`` such that ``x == n * (n + 1) / 2``.

    Parameters
    ----------
    x : int or float
        The number to check

    Returns
    -------
    : bool
        True if the number is a triangular number, False otherwise.
    """
    # number must be an integer, but we also accept floats that are actually
    # integers
    if not (isinstance(x, numbers.Integral) or x.is_integer()) or x < 1:
        return False, -1

    # use quadratic formula to solve
    n = 0.5 * (np.sqrt(1 + 8 * x) - 1)
    return n.is_integer(), int(n)


def unvech(a):
    """Recover a symmetric matrix from its half-vectorization.

    Parameters
    ----------
    a : np.ndarray, shape (m,)
        The half-vectorization of some square matrix. The size must satisfy m =
        n * (n + 1) / 2 for some integer n >= 1.

    Returns
    -------
    : np.ndarray, shape (n, n)
        The unvectorized matrix.
    """
    a = np.array(a, copy=False)
    assert a.ndim == 1
    s = a.shape[0]

    # r = 0.5 * (np.sqrt(1 + 8 * s) - 1)
    # n = np.rint(r).astype(int)
    # assert np.isclose(r, n),
    triangular, n = is_triangular_number(s)
    assert triangular, "vector is not the vech of a square matrix."

    # fill in the upper triangle
    A = np.zeros((n, n))
    idx = np.triu_indices(n)
    A[idx] = a

    # fill in the lower triangle
    idxl = np.tril_indices(n, k=-1)
    A[idxl] = A.T[idxl]

    return A


def skew3(v):
    """Form a skew-symmetric matrix from a 3-vector.

    This is such that ``skew3(a) @ b == cross(a, b)``, where ``cross`` is the
    cross product.

    Parameters
    ----------
    v : np.ndarray, shape (3,)
        The vector to make skew-symmetric.

    Returns
    -------
    : np.ndarray, shape (3, 3)
        The skew-symmetric matrix corresponding to ``v``.
    """
    assert v.shape == (3,)
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def lift3(x):
    """Lift a 3-vector x such that A @ x = lift3(x) @ vech(A) for symmetric A."""
    assert x.shape == (3,)
    # fmt: off
    return np.array([
        [x[0], x[1], x[2],    0,    0,    0],
        [   0, x[0],    0, x[1], x[2],    0],
        [   0,    0, x[0],    0, x[1], x[2]]
    ])
    # fmt: on


def lift6(V):
    """Lift a twist V such that M @ V = lift6(V) @ θ.

    M is the spatial mass matrix and θ is the corresponding inertial parameter
    vector.
    """
    a = V.linear
    b = V.angular
    # fmt: off
    return np.block([
        [np.zeros((3, 1)), -skew3(a),         lift3(b)],
        [      a[:, None],  skew3(b), np.zeros((3, 6))]])
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


def validation_rmse(Ys, ws, θ, W=None):
    """Compute root mean square wrench error on a validation set."""
    if W is None:
        W = np.eye(6)
    error = Ys @ θ - ws
    # square = np.sum(error @ W * error, axis=1)
    square = np.sum(error**2, axis=1)
    assert square.shape[0] == ws.shape[0]
    mean = np.mean(square)
    root = np.sqrt(mean)
    return root


def clean_transform(rotation, translation, dim=3):
    if rotation is None:
        rotation = np.eye(dim)
    else:
        rotation = np.array(rotation)
        assert rotation.shape == (dim, dim)

    if translation is None:
        translation = np.zeros(dim)
    else:
        translation = np.array(translation)
        assert translation.shape == (dim,)

    return rotation, translation


def transform_matrix(rotation=None, translation=None, dim=3):
    """Construct a transformation matrix from a rotation and translation."""
    rotation, translation = clean_transform(rotation, translation, dim=dim)
    T = np.eye(dim + 1)
    T[:dim, :dim] = rotation
    T[:dim, -1] = translation
    return T


def transform_matrix_inv(rotation=None, translation=None, dim=3):
    """Construct the inverse transformation matrix from a rotation and
    translation."""
    rotation, translation = clean_transform(rotation, translation, dim=dim)
    return transform_matrix(
        rotation=rotation.T, translation=-rotation.T @ translation
    )


def box_vertices(half_extents):
    """Generate the vertices of a multi-dimensional box.

    Parameters
    ----------
    half_extents : np.ndarray, shape (dim,)
        The half extents of the box. The shape indicates the box's dimension.

    Returns
    -------
    : np.ndarray, shape (2**dim, dim)
        The vertices of the box, centered at the origin and aligned with axes.
    """
    dim = len(half_extents)
    combos = np.array([c for c in itertools.product([1, -1], repeat=dim)])
    return combos * half_extents


def contact_jacobian(point):
    """Compute the contact Jacobian for a point.

    Parameters
    ----------
    point : np.ndarray, shape (3,)
        The contact point.

    Returns
    -------
    : np.ndarray, shape (6, 3)
        The contact Jacobian matrix, which the contact force to the applied
        contact wrench.
    """
    return np.vstack((skew3(point), np.eye(3)))
