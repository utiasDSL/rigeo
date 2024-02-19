import numpy as np


def positive_definite_distance(A, B):
    """Geodesic distance between two symmetric positive definite matrices
    :math:`A` and :math:`B`.

    This metric is coordinate-frame invariant. See :cite:t:`lee2019geometric`.
    for more details.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        A symmetric positive definite matrix.
    B : np.ndarray, shape (n, n)
        A symmetric positive definite matrix.

    Returns
    -------
    : float, non-negative
        The geodesic distance between :math:`A` and :math:`B`.
    """
    assert A.shape == B.shape, "Matrices must have the same shape."
    C = np.linalg.solve(A, B)
    eigs = np.linalg.eigvals(C)
    return np.sqrt(0.5 * np.sum(np.log(eigs) ** 2))
