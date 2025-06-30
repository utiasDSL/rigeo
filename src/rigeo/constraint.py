import numpy as np
import cvxpy as cp


def _shape_schur_var(X):
    """Shape a variable into a 2-ndim expression."""
    if np.isscalar(X) or (isinstance(X, cp.Expression) and X.size == 1):
        X = cp.reshape(X, (1, 1), order="C")
    elif X.ndim == 1:
        X = cp.reshape(X, (X.shape[0], 1), order="C")
    return X


def schur(A, B, C):
    """Construct the Schur complement matrix

    .. math::
        \\begin{bmatrix} \\boldsymbol{A} & \\boldsymbol{B} \\\\ \\boldsymbol{B}^T & \\boldsymbol{C} \\end{bmatrix}

    Parameters
    ----------
    A : float or np.ndarray or cp.Expression
    B : float or np.ndarray or cp.Expression
    C : float or np.ndarray or cp.Expression

    Returns
    -------
    : cp.Expression
        The Schur complement matrix.
    """
    A = _shape_schur_var(A)
    B = _shape_schur_var(B)
    C = _shape_schur_var(C)
    return cp.bmat([[A, B], [B.T, C]])


def pim_sum_vec_matrices():
    """Generate the matrices A_i such that J == sum(A_i * θ_i)"""
    As = [np.zeros((4, 4)) for _ in range(10)]
    As[0][3, 3] = 1  # mass

    # hx
    As[1][0, 3] = 1
    As[1][3, 0] = 1

    # hy
    As[2][1, 3] = 1
    As[2][3, 1] = 1

    # hz
    As[3][2, 3] = 1
    As[3][3, 2] = 1

    # Ixx
    As[4][0, 0] = -0.5
    As[4][1, 1] = 0.5
    As[4][2, 2] = 0.5

    # Ixy
    As[5][0, 1] = -1
    As[5][1, 0] = -1

    # Ixz
    As[6][0, 2] = -1
    As[6][2, 0] = -1

    # Iyy
    As[7][0, 0] = 0.5
    As[7][1, 1] = -0.5
    As[7][2, 2] = 0.5

    # Iyz
    As[8][1, 2] = -1
    As[8][2, 1] = -1

    # Izz
    As[9][0, 0] = 0.5
    As[9][1, 1] = 0.5
    As[9][2, 2] = -0.5

    return As


# TODO not sure I totally like the semantics of this
def pim_must_equal_vec(θ):
    """Generate a cvxpy expression that converts a parameter vector to
    pseudo-inertia matrix.

    Parameters
    ----------
    θ : np.ndarray or cp.Expression, shape (10,)
        The inertial parameter vector.

    Returns
    -------
    : cp.Expression, shape (4, 4)
        The corresponding pseudo-inertia matrix.
    """
    assert θ.shape == (10,)
    return cp.sum([A * p for A, p in zip(pim_sum_vec_matrices(), θ)])


def pim_must_equal_param_var(param_var, eps):
    """Generate cvxpy constraints that ensure the pseudo-inertia matrix equals
    the inertial parameter variable.

    If the inertial parameter variable is of shape ``(4, 4)``, we treat is as a
    pseudo-inertia matrix directly. If it is of shape ``(10,)``, then we treat
    it is an inertial parameter vector.

    Parameters
    ----------
    param_var : cp.Expression, shape (4, 4) or (10,)
        The inertial parameter variable.
    eps : float, non-negative
        Enforce ``J - eps * np.eye(4)`` is positive semidefinite.

    Returns
    -------
    : tuple
        A tuple ``(J, constraints)``, where ``J`` is the pseudo-inertia matrix
        and ``constraints`` is a list of cvxpy constraints.
    """

    assert eps >= 0
    if param_var.shape == (4, 4):
        J = param_var
    elif param_var.shape == (10,):
        J = pim_must_equal_vec(param_var)
    else:
        raise ValueError(f"Parameter variable has unexpected shape {param_var.shape}")

    return J, pim_psd(J, eps=eps)


def pim_psd(J, eps=0):
    """Generate cvxpy constraints that ensure J is symmetric positive definite.

    Parameters
    ----------
    J : cp.Expression, shape (4, 4)
        The pseudo-inertia matrix.
    eps : float, non-negative
        Enforce ``J - eps * np.eye(4)`` is positive semidefinite.

    Returns
    -------
    : list
        A list of cvxpy constraints to enforce that ``J`` is symmetric positive
        (semi-)definite.
    """
    assert J.shape == (4, 4)
    assert eps >= 0
    return [J == J.T, J >> eps * np.eye(4)]
