"""Claim: aa^T <= bb^T + cc^T --> a = α1 + b * α2 * c"""
import numpy as np
from scipy.linalg import sqrtm
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from spatialmath.base import rot2
from scipy.optimize import minimize

import IPython

np.random.seed(0)


def rank_one_problem():
    for i in range(100):
        L = np.random.random((3, 3))
        A = L @ L.T
        A /= A[2, 2]
        A *= 0.5

        e, v = np.linalg.eig(A[:2, :2])
        s = np.sqrt(e)
        b = s[0] * v[:, 0]
        c = s[1] * v[:, 1]
        assert np.allclose(np.outer(b, b) + np.outer(c, c), A[:2, :2])
        a = A[:2, 2]

        α1 = cp.Variable(1)
        α2 = cp.Variable(1)

        objective = cp.Minimize([0])
        constraints = [
            a == α1 * b + α2 * c,
            -α1 - α2 <= 1.0,
            α1 + α2 <= 1.0,
            -α1 + α2 <= 1.0,
            α1 - α2 <= 1.0,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != "optimal":
            print(problem.status)
            IPython.embed()
            break


def generate_general_sample(n, max_tries=100):
    """Generate two positive semidefinite matrices X and Y such that Y @ Y.T -
    X @ X.T is positive semidefinite.
    """
    for _ in range(max_tries):
        X = np.random.random((n, n))
        Y = np.random.random((n, n))
        if np.min(np.linalg.eigvals(Y @ Y.T - X @ X.T)) >= 0:
            return X, Y
    raise ValueError("Failed to generate sample.")


def general_problem():
    # NOTE: this does not work, but this may be because I cannot specify the
    # vectors x1 and x2 beforehand; instead it is only the case that there
    # *exist* such vectors
    for i in range(100):
        X, Y = generate_general_sample(2)
        Z = np.block([[Y @ Y.T, X], [X.T, np.eye(2)]])

        ey, vy = np.linalg.eig(Y @ Y.T)
        sy = np.sqrt(ey)
        # y1 = Y[:2, 0]
        # y2 = Y[:2, 1]
        y1 = sy[0] * vy[:, 0]
        y2 = sy[1] * vy[:, 1]

        ex, vx = np.linalg.eig(X @ X.T)
        sx = np.sqrt(ex)
        # x1 = X[:2, 0]
        # x2 = X[:2, 1]
        x1 = sx[0] * vx[:, 0]
        x2 = sx[1] * vx[:, 1]

        α11 = cp.Variable(1)
        α12 = cp.Variable(1)
        α21 = cp.Variable(1)
        α22 = cp.Variable(1)

        objective = cp.Minimize([0])
        # fmt: off
        constraints = [
            x1 == α11 * y1 + α12 * y2,
            x2 == α21 * y1 + α22 * y2,

            -α11 - α12 <= 1.0,
            α11 + α12 <= 1.0,
            -α11 + α12 <= 1.0,
            α11 - α12 <= 1.0,

            -α21 - α22 <= 1.0,
            α21 + α22 <= 1.0,
            -α21 + α22 <= 1.0,
            α21 - α22 <= 1.0,
        ]
        # fmt: on
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != "optimal":
            print(problem.status)
            IPython.embed()
            break


def build_candidate(α1, α2, β1, β2):
    assert np.abs(α1) + np.abs(α2) <= 1
    assert np.abs(β1) + np.abs(β2) <= 1

    a = np.array([α1, α2])
    b = np.array([β1, β2])
    return np.outer(a, a) + np.outer(b, b)


def plot_ellipse(ax, X, color=None):
    """X is 2x2 psd matrix."""
    assert X.shape == (2, 2)
    e, V = np.linalg.eig(X)
    assert np.min(e) >= 0

    angle = np.arccos(V[:, 0] @ [1, 0])

    ell = Ellipse(
        xy=[0, 0],
        width=2 * e[0],
        height=2 * e[1],
        angle=np.rad2deg(angle),
        color=color,
        fill=False,
    )
    ax.add_patch(ell)


def plot_diamond(ax, X, color=None):
    assert X.shape == (2, 2)
    e, V = np.linalg.eig(X)
    assert np.min(e) >= 0

    a = e[0] * V[:, 0]
    b = e[1] * V[:, 1]
    x = np.array([a[0], b[0], -a[0], -b[0], a[0]])
    y = np.array([a[1], b[1], -a[1], -b[1], a[1]])
    plt.plot(x, y, "--", color=color)


def match_matrix(T):
    def points(x):
        return x[:2], x[2:]

    def mat(x):
        x1, x2 = points(x)
        return np.outer(x1, x1) + np.outer(x2, x2)

    def cost(x):
        X = mat(x)
        return np.sum((T - X) ** 2)

    def ineq_con(x):
        return np.array(
            [1 - np.abs(x[0]) - np.abs(x[1]), 1 - np.abs(x[2]) - np.abs(x[3])]
        )

    constraints = [{"type": "ineq", "fun": ineq_con}]
    x0 = np.array([1, 0, 0, 1])
    res = minimize(cost, x0=x0, constraints=constraints)
    if not res.success:
        print("optimization failed")
        IPython.embed()

    x1, x2 = points(res.x)
    X = mat(res.x)

    assert np.sum(np.abs(x1)) <= 1
    assert np.sum(np.abs(x2)) <= 1

    return X, x1, x2


def visualize():
    # visualize a specific example
    plt.figure()
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.grid()

    ax = plt.gca()
    ax.set_aspect("equal")

    # baseline circle
    plot_ellipse(ax, np.eye(2), color="k")
    plot_diamond(ax, np.eye(2), color="k")

    C1 = rot2(3 * np.pi / 8)
    C2 = rot2(-1 * np.pi / 8)
    t1 = C1 @ [1, 0]
    t2 = 0.95 * C2 @ [1, 0]
    T = np.outer(t1, t1) + np.outer(t2, t2)
    et = np.sort(np.linalg.eigvals(T))

    X, x1, x2 = match_matrix(T)

    plot_ellipse(ax, T, color="b")
    plot_ellipse(ax, X, color="r")
    plt.plot([x1[0], x2[0]], [x1[1], x2[1]], "o", color="r")

    plt.show()


def optimize_random_problems():
    for i in range(1000):
        # generate random p.s.d. matric with eigmax <= 1
        X = 2 * np.random.random((2, 2)) - 1
        Y = X @ X.T
        
        # normalize to ensure max eigenvalue is at most 1
        e_max = np.max(np.linalg.eigvals(Y))
        if e_max > 1:
            Y /= e_max

        # match the matrix using outer product of two vectors with l1 norms <= 1
        S, s1, s2 = match_matrix(Y)
        
    print("success!")
    # IPython.embed()


# visualize()
# optimize_random_problems()

# generate random p.s.d. matric with eigmax <= 1
X = 2 * np.random.random((2, 2)) - 1
Y = X @ X.T

# normalize to ensure max eigenvalue is at most 1
e_max = np.max(np.linalg.eigvals(Y))
if e_max > 1:
    Y /= e_max

M = np.block([[Y, X], [X, np.zeros((2, 2))]])
IPython.embed()
