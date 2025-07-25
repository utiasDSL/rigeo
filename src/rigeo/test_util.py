"""Utilities for testing."""
import numpy as np


def allclose_unordered(A, B, **kwargs):
    """Check if two matrices are close, ignoring the order of rows.

    Takes all the same keyword arguments as ``numpy.isclose``.

    Parameters
    ----------
    A : np.ndarray
        The first matrix to compare.
    B : np.ndarray, same shape as A
        The second matrix to compare.

    Returns
    -------
    : bool
        ``True`` if the matrices are close, ``False`` otherwise.
    """
    assert A.shape == B.shape
    n = A.shape[0]
    B_checked = np.zeros(n, dtype=bool)
    for i in range(n):
        a = A[i, :]

        # True where ``a`` matches a row of ``B``
        equal = np.isclose(a, B, **kwargs).all(axis=1)

        # find any locations where ``a`` matches a row of ``B`` that has not
        # been checked yet
        test = np.logical_and(equal, ~B_checked)

        # If any such locations exist, mark the first one as checked and
        # continue. If none exist, the matrices do not match, so return False.
        idx = np.argmax(test)
        if test[idx]:
            B_checked[idx] = True
        else:
            return False
    return True
