import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
from scipy.spatial import ConvexHull


# def cuboid_vertices(half_extents):
#     """Vertices of a cuboid with given half extents."""
#     x, y, z = half_extents
#     return np.array(
#         [
#             [x, y, z],
#             [x, y, -z],
#             [x, -y, z],
#             [x, -y, -z],
#             [-x, y, z],
#             [-x, y, -z],
#             [-x, -y, z],
#             [-x, -y, -z],
#         ]
#     )


class AxisAlignedBox:
    def __init__(self, half_extents, center=None):
        self.half_extents = np.array(half_extents)
        assert self.half_extents.shape == (3,)
        if center is None:
            center = np.zeros(3)
        self.center = np.array(center)

    @classmethod
    def cube(cls, half_extent, center=None):
        half_extents = half_extent * np.ones(3)
        return cls(half_extents, center=center)

    @classmethod
    def from_side_lengths(cls, side_lengths, center=None):
        return cls(0.5 * side_lengths, center=center)

    @classmethod
    def from_two_vertices(cls, v1, v2):
        center = 0.5 * (v1 + v2)
        half_extents = 0.5 * (np.maximum(v1, v2) - np.minimum(v1, v2))
        return cls(half_extents, center=center)

    @property
    def side_lengths(self):
        return 2 * self.half_extents

    @property
    def vertices(self):
        x, y, z = self.half_extents
        return (
            np.array(
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
            + self.center
        )

    def random_points(self, shape=1):
        """Generate a set of random points contained in the box."""
        d = self.half_extents.shape[0]
        if type(shape) is int:
            shape = (shape,)
        shape = shape + (d,)

        points = self.half_extents * (2 * np.random.random(shape) - 1) + self.center
        if shape == (1, d):
            return points.flatten()
        return points

    def contains(self, points):
        """Test if the box contains a set of points."""
        points = np.atleast_2d(points)
        return (np.abs(points - self.center) <= self.half_extents).all(axis=1)


def convex_hull(points):
    """Get the vertices of the convex hull of a set of points."""
    hull = ConvexHull(points)
    return points[hull.vertices, :]


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
        """Return True if x is inside the ellipsoid, False otherwise."""
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

    See Convex Optimization by Boyd & Vandenberghe, sec. 8.4.1.

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
