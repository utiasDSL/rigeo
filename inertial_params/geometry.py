import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm


def cuboid_vertices(half_extents):
    """Vertices of a cuboid with given half extents."""
    x, y, z = half_extents
    return np.array(
        [
            [x, y, z],
            [x, y, -z],
            [x, -y, z],
            [x, -y, -z],
            [-x, y, z],
            [-x, y, -z],
            [-x, -y, z],
            [-x, -y, -z],
        ]
    )


def convex_hull(points):
    # TODO
    pass


class Ellipsoid:
    """Ellipsoid with a variety of representations."""

    def __init__(self, Einv, c):
        self.Einv = Einv
        self.c = c

    @classmethod
    def sphere(cls, radius):
        Einv = np.eye(3) / radius**2
        return cls(Einv=Einv, c=np.zeros(3))

    @classmethod
    def from_Ab(cls, A, b):
        Einv = A.T @ A
        c = np.linalg.solve(Einv, -A.T @ b)
        return cls(Einv=Einv, c=c)

    @classmethod
    def from_Q(cls, Q):
        Einv = -Q[:3, :3]
        q = Q[:3, 3]
        c = np.linalg.solve(Einv, q)
        return cls(Einv=Einv, c=c)

    @property
    def A(self):
        return sqrtm(self.Einv)

    @property
    def b(self):
        return np.linalg.solve(self.A.T, -self.Einv @ self.c)

    @property
    def Q(self):
        Q = np.zeros((4, 4))
        Q[:3, :3] = -self.Einv
        Q[:3, 3] = self.Einv @ self.c
        Q[3, :3] = Q[:3, 3]
        Q[3, 3] = 1 - self.c @ self.Einv @ self.c
        return Q

    def contains(self, x):
        p = x - self.c
        return p @ self.Einv @ p <= 1

    def transform(self, C=None, r=None):
        if C is None:
            C = np.eye(3)
        if r is None:
            r = np.zeros(3)
        Einv = C @ self.Einv @ C.T
        c = self.c + r
        return Ellipsoid(Einv=Einv, c=c)


def cube_bounding_ellipsoid(h):
    """Bounding ellipsoid (sphere) of a cube with half length h.

    Returns the ellipsoid.
    """
    r = np.linalg.norm([h, h, h])
    return Ellipsoid.sphere(r)


# TODO currently this only works with non-degenerate ellipsoids
def minimum_bounding_ellipsoid(points):
    """Compute the minimum bounding ellipsoid for a set of points.

    See Convex Optimization by Boyd & Vandenberghe, sec. 8.4.1

    Returns the ellipsoid.
    """
    # ellipsoid is parameterized as ||Ax + b|| <= 1 for the opt problem
    A = cp.Variable((3, 3), PSD=True)
    b = cp.Variable(3)
    objective = cp.Minimize(-cp.log_det(A))
    constraints = [cp.norm2(A @ x + b) <= 1 for x in points]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    return Ellipsoid.from_Ab(A=A.value, b=b.value)
