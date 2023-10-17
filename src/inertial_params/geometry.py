import functools

import numpy as np
import cvxpy as cp
import cdd
from scipy.linalg import sqrtm, orth
from scipy.spatial import ConvexHull


# TODO this cannot be made general for the ellipsoid and the convex hull
# without a lot of cludge
# def lowranknoproblem(f, rcond=None):
#     @functools.wraps(f)
#     def wrapper(points, **kwargs):
#         R = orth(points.T, rcond=rcond)
#         P = points @ R
#         H = f(P, **kwargs)
#         pass
#     return wrapper


def convex_hull(points, rcond=None):
    """Get the vertices of the convex hull of a set of points.

    Parameters
    ----------
    points :
        A :math:`n\\times d` array of points, where :math:`n` is the number of
        points and `d` is the dimension. Note that the points must be full rank
        (i.e., they must span :math:`\\mathbb{R}^d`.

    Returns
    -------
    :
        The :math:`m\\times d` array of vertices of the convex hull that fully
        contains the set of points.
    """
    # rowspace
    R = orth(points.T, rcond=rcond)

    # project onto the rowspace
    # this allows us to handle degenerate sets of points that live in a
    # lower-dimensional subspace than R^d
    P = points @ R

    # find the hull
    hull = ConvexHull(P)
    H = P[hull.vertices, :]

    # unproject
    return H @ R.T


def polyhedron_span_to_face_form(vertices):
    """Convert a set of vertices to a set of linear inequalities A <= b."""
    # span form
    n = vertices.shape[0]
    Smat = cdd.Matrix(np.hstack((np.ones((n, 1)), vertices)))
    Smat.rep_type = cdd.RepType.GENERATOR

    # polyhedron
    poly = cdd.Polyhedron(Smat)

    # general face form is Ax <= b, which cdd stores as one matrix [b -A]
    Fmat = poly.get_inequalities()
    F = np.array([Fmat[i] for i in range(Fmat.row_size)])
    b = F[:, 0]
    A = -F[:, 1:]
    return A, b


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

    def grid(self, n):
        L = self.center - self.half_extents
        U = self.center + self.half_extents

        x = np.linspace(L[0], U[0], n)
        y = np.linspace(L[1], U[1], n)
        z = np.linspace(L[2], U[2], n)

        points = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    points.append([x[i], y[j], z[k]])
        return np.array(points)


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
        Einv = A @ A
        c = np.linalg.solve(A, -b)
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
        return -self.A @ self.c

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
    """Minimum-volume bounding ellipsoid (sphere) of a cube with half length h.

    Returns the ellipsoid.
    """
    r = np.linalg.norm([h, h, h])
    return Ellipsoid.sphere(r)


def cube_inscribed_ellipsoid(h):
    """Maximum-volume inscribed ellipsoid (sphere) of a cube with half length h.

    Returns the ellipsoid.
    """
    return Ellipsoid.sphere(h)


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


def maximum_inscribed_ellipsoid(A, b):
    """Compute the maximum inscribed ellipsoid for an inequality-form polyhedron.

    See Convex Optimization by Boyd & Vandenberghe, sec. 8.4.2.

    Returns the ellipsoid.
    """
    B = cp.Variable((3, 3), PSD=True)
    c = cp.Variable(3)
    n = b.shape[0]

    objective = cp.Maximize(cp.log_det(B))
    constraints = [cp.norm2(B @ A[i, :]) + A[i, :] @ c <= b[i] for i in range(n)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    E = B.value @ B.value
    return Ellipsoid(Einv=np.linalg.inv(E), c=c.value)
