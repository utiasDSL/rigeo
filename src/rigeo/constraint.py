import numpy as np
import cvxpy as cp


def schur(A, b, c):
    """Construct the matrix [[A, b], [b.T, c]].

    Parameters must be cvxpy expressions.
    """
    # cvxpy variables
    b = cp.reshape(b, (b.shape[0], 1))
    c = cp.reshape(c, (1, 1))
    return cp.bmat([[A, b], [b.T, c]])


def _pim_sum_vec_matrices():
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


def pim_must_equal_vec(θ):
    return cp.sum([A * p for A, p in zip(_pim_sum_vec_matrices(), θ)])


def pim_must_equal_param_var(param_var, eps):
    assert eps >= 0
    if param_var.shape == (4, 4):
        J = param_var
    elif param_var.shape == (10,):
        J = pim_must_equal_vec(param_var)
    else:
        raise ValueError(f"Parameter variable has unexpected shape {param_var.shape}")

    return J, [J == J.T, J >> eps * np.eye(4)]
