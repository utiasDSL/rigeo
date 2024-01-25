"""Polyhedral and ellipsoidal geometry."""
import numpy as np
import cvxpy as cp
import cdd
from scipy.linalg import sqrtm, orth
from scipy.spatial import ConvexHull

from inertial_params.random import random_weight_vectors


class ConvexPolyhedron:
    """A convex polyhedron in ``dim`` dimensions.

    Parameters
    ----------
    vertices : np.ndarray, shape (nv, dim)
        The extremal points of the polyhedron.
    prune_vertices : bool
        If ``True``, the vertices will be pruned to eliminate any non-extremal
        points.

    Attributes
    ----------
    vertices : np.ndarray, shape (nv, dim)
        The extremal points of the polyhedron.
    A : np.ndarray
        The matrix part of the face (half-space) form of the polyhedron,
        :math:`\\{p\\in\\mathbb{R}^{dim}\\mid Ap\\leq b \\}`
    b : np.ndarray
        The vector part of the face form of the polyhedron.
    """
    def __init__(self, vertices, prune_vertices=False):
        if prune_vertices:
            vertices = convex_hull(vertices)
        self.vertices = np.array(vertices)
        self.A, self.b = polyhedron_span_to_face_form(vertices)

    def __repr__(self):
        return f"ConvexPolyhedron(vertices={self.vertices})"

    @property
    def nv(self):
        """The number of vertices."""
        return self.vertices.shape[0]

    @property
    def dim(self):
        """The dimension of the ambient space."""
        return self.vertices.shape[1]

    def contains(self, points, tol=1e-8):
        """Test if the polyhedron contains a set of points.

        Parameters
        ----------
        points : np.ndarray, shape (n, self.dim)
            The points to check.
        tol : float, non-negative
            The numerical tolerance for membership.

        Returns
        -------
        : np.ndarray, shape (n,)
            Boolean array where each entry is ``True`` if the polyhedron
            contains the corresponding point and ``False`` otherwise.
        """
        points = np.atleast_2d(points)
        c = self.A @ points.T <= np.tile(self.b, (points.shape[0], 1)).T + tol
        return np.array(np.product(c, axis=0), dtype=bool)

    def random_points(self, shape=1):
        """Generate random points contained in the polyhedron.

        Parameters
        ----------
        shape : int or tuple
            The shape of the set of points to be returned.

        Returns
        -------
        : np.ndarray, shape ``shape + (self.dim,)``
            The random points.
        """
        if type(shape) is int:
            shape = (shape,)
        shape = shape + (self.nv,)
        w = random_weight_vectors(shape)
        return w @ self.vertices

    def aabb(self):
        """Generate an axis-aligned box that bounds the polyhedron.

        Returns
        -------
        : AxisAlignedBox
            The axis-aligned bounding box.
        """
        return AxisAlignedBox.from_points_to_bound(self.vertices)

    def grid(self, n):
        """Generate a regular grid inside the polyhedron.

        The approach is to compute the axis-aligned bounding box, generate a
        grid for that, and then discard any points not inside the actual
        polyhedron.

        Parameters
        ----------
        n : int
            The maximum number of points along each dimension.

        Returns
        -------
        : np.ndarray, shape (N, self.dim)
            The points contained in the grid.
        """
        box_grid = self.aabb().grid(n)
        contained = self.contains(box_grid)
        return box_grid[contained, :]


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
    assert points.ndim == 2
    if points.shape[0] <= 1:
        return points

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


# TODO deprecate in favour of ConvexPolyhedron.grid
def polyhedron_grid(vertices, n):
    """Generate a regular grid inside a convex polyhedron.

    The approach is to compute the axis-aligned bounding box, generate a grid
    for that, and then discard any points not inside the actual polyhedron.

    Parameters
    ----------
    vertices :
        The vertices of the polyhedron.
    n :
        The maximum number of points along each dimension.

    Returns
    -------
    :
        An array of shape ``(N, 3)`` representing the ``N`` points in the grid.
    """
    bounding_box = AxisAlignedBox.from_points_to_bound(vertices)
    box_grid = bounding_box.grid(n)
    A, b = polyhedron_span_to_face_form(vertices)
    contains = (A @ box_grid.T <= b[:, None]).T.all(axis=1)
    return box_grid[contains, :]


class AxisAlignedBox:
    """A box aligned with the x, y, z axes.

    Parameters
    ----------
    half_extents :
        The (x, y, z) half extents of the box. The half extents are each half
        the length of the corresponding side lengths.
    center :
        The center of the box. Defaults to the origin.

    Attributes
    ----------
    half_extents :
        The (x, y, z) half extents of the box.
    center :
        The center of the box.
    """

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

    @classmethod
    def from_points_to_bound(cls, points):
        v_min = np.min(points, axis=0)
        v_max = np.max(points, axis=0)
        return cls.from_two_vertices(v_min, v_max)

    def __repr__(self):
        return f"AxisAlignedBox(center={self.center}, half_extents={self.half_extents})"

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

    def contains(self, points, tol=1e-8):
        """Test if the box contains a set of points."""
        points = np.atleast_2d(points)
        return (np.abs(points - self.center) <= self.half_extents + tol).all(axis=1)

    def grid(self, n):
        """Generate a set of points evenly spaced in the box.

        Parameters
        ----------
        n : int
            The number of points in each of the three dimensions.

        Returns
        -------
        :
            An array of points with shape ``(n**3, 3)``.
        """
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


# TODO rename c -> center, possibly rename Einv also
class Ellipsoid:
    """Ellipsoid with a variety of representations."""

    def __init__(self, Einv, c):
        assert Einv.ndim == 2
        assert Einv.shape[0] == Einv.shape[1]
        assert Einv.shape[0] == c.shape[0]

        self.Einv = Einv
        self.c = c
        self.dim = c.shape[0]
        self.rank = np.linalg.matrix_rank(Einv)

    def __repr__(self):
        return f"Ellipsoid(center={self.c}, Einv={self.Einv})"

    @classmethod
    def sphere(cls, radius, center=None, dim=None):
        """Construct a sphere.

        Parameters
        ----------
        radius : float
            Radius of the sphere.
        center : iterable
            Optional center point of the sphere.
        dim : int
            Dimension of the sphere. Only required if ``center`` not provided,
            in which case the sphere is centered at the origin. If neither
            ``center`` or ``dim`` is provided, defaults to a 3D sphere at the
            origin.
        """
        if center is None:
            if dim is None:
                dim = 3
            center = np.zeros(dim)
        else:
            center = np.array(center)
        assert center.shape == (dim,)

        Einv = np.eye(dim) / radius**2
        return cls(Einv=Einv, c=center)

    @classmethod
    def from_half_extents(cls, half_extents, center=None):
        """Construct an ellipsoid from its half extents, which are the lengths
        of the semi-major axes.

        Parameters
        ----------
        half_extents : np.ndarray, shape (n,)
            The lengths of the semi-major axes, where ``n`` is the dimension.
        center : np.ndarray, shape (n,)
            Optional center point of the ellipsoid.
        """
        half_extents = np.array(half_extents)
        Einv = np.diag(1. / half_extents**2)
        if center is None:
            center = np.zeros_like(half_extents)
        return cls(Einv=Einv, c=center)

    @classmethod
    def from_Ab(cls, A, b, rcond=None):
        Einv = A @ A

        # use least squares instead of direct solve in case we have a
        # degenerate ellipsoid
        c = np.linalg.lstsq(A, -b, rcond=rcond)[0]
        return cls(Einv=Einv, c=c)

    @classmethod
    def from_Q(cls, Q):
        assert Q.shape[0] == Q.shape[1]
        dim = Q.shape[0] - 1
        Einv = -Q[:dim, :dim]
        q = Q[:dim, dim]
        c = np.linalg.solve(Einv, q)
        return cls(Einv=Einv, c=c)

    def axes(self):
        """Compute axes directions and semi-axes values."""
        e, V = np.linalg.eig(self.Einv)
        r = 1 / np.sqrt(e)
        return r, V

    @property
    def A(self):
        """``A`` from ``(A, b)`` representation of the ellipsoid.

        .. math::
           \\mathcal{E} = \\{x\\in\\mathbb{R}^d \\mid \\|Ax+b\\|^2\\leq 1\\}
        """
        return sqrtm(self.Einv)

    @property
    def b(self):
        """``b`` from ``(A, b)`` representation of the ellipsoid.

        .. math::
           \\mathcal{E} = \\{x\\in\\mathbb{R}^d \\mid \\|Ax+b\\|^2\\leq 1\\}
        """
        return -self.A @ self.c

    @property
    def Q(self):
        """Q representation of the ellipsoid.

        .. math::
           \\mathcal{E} = \\{x\\in\\mathbb{R}^d \\mid \\tilde{x}^TQ\\tilde{q}\\geq 0\\}
        """

        Q = np.zeros((self.dim + 1, self.dim + 1))
        Q[: self.dim, : self.dim] = -self.Einv
        Q[: self.dim, self.dim] = self.Einv @ self.c
        Q[self.dim, : self.dim] = Q[: self.dim, self.dim]
        Q[self.dim, self.dim] = 1 - self.c @ self.Einv @ self.c
        return Q

    def degenerate(self):
        """Check if the ellipsoid is degenerate.

        This means that it has zero volume, and lives in a lower dimension than
        the ambient one.

        Returns
        -------
        : bool
            Returns ``True`` if the ellipsoid is degenerate, ``False`` otherwise.
        """
        return self.rank < self.dim

    def contains(self, points, tol=1e-8):
        """Check if points are contained in the ellipsoid.

        Parameters
        ----------
        points : iterable
            Points to check. May be a single point or a list or array of points.
        tol : float, non-negative
            Numerical tolerance for qualifying as inside the ellipsoid.

        Returns
        -------
        :
            Given a single point, return ``True`` if the point is contained in
            the ellipsoid, or ``False`` if not. For multiple points, return a
            boolean array with one value per point.
        """
        points = np.array(points)
        if points.ndim == 1:
            p = points - self.c
            return p @ self.Einv @ p <= 1 + tol
        elif points.ndim == 2:
            ps = points - self.c
            return np.array([p @ self.Einv @ p <= 1 + tol for p in ps])
        else:
            raise ValueError(
                f"points must have 1 or 2 dimensions, but has {points.ndim}."
            )

    def transform(self, C=None, r=None):
        """Apply an affine transform to the ellipsoid.

        Parameters
        ----------
        C :
            Rotation matrix.
        r :
            Translation vector.

        Returns
        -------
        : Ellipsoid
            A new ellipsoid that is an affine transform of this one.
        """
        if C is None and r is None:
            return Ellipsoid(Einv=self.Einv.copy(), c=self.c.copy())

        if C is None:
            dim = r.shape[0]
            C = np.eye(dim)
        if r is None:
            dim = C.shape[0]
            r = np.zeros(dim)

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


def minimum_bounding_ellipsoid(points, rcond=None):
    """Compute the minimum bounding ellipsoid for a set of points.

    See Convex Optimization by Boyd & Vandenberghe, sec. 8.4.1.

    Returns the ellipsoid.
    """
    # rowspace
    R = orth(points.T, rcond=rcond)

    # project onto the rowspace
    # this allows us to handle degenerate sets of points that live in a
    # lower-dimensional subspace than R^d
    P = points @ R

    dim = P.shape[1]

    # ellipsoid is parameterized as ||Ax + b|| <= 1 for the opt problem
    A = cp.Variable((dim, dim), PSD=True)
    b = cp.Variable(dim)

    objective = cp.Minimize(-cp.log_det(A))
    constraints = [cp.norm2(A @ x + b) <= 1 for x in P]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    # unproject back into the original space
    A = R @ A.value @ R.T
    b = R @ b.value
    return Ellipsoid.from_Ab(A=A, b=b, rcond=rcond)


# TODO this also does not handle degeneracy
def _maximum_inscribed_ellipsoid_inequality_form(A, b):
    """Compute the maximum inscribed ellipsoid for an inequality-form
    polyhedron P = {x | Ax <= b}.

    See Convex Optimization by Boyd & Vandenberghe, sec. 8.4.2.

    Returns the ellipsoid.
    """

    dim = A.shape[1]
    n = b.shape[0]

    B = cp.Variable((dim, dim), PSD=True)
    c = cp.Variable(dim)

    objective = cp.Maximize(cp.log_det(B))
    constraints = [cp.norm2(B @ A[i, :]) + A[i, :] @ c <= b[i] for i in range(n)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    E = B.value @ B.value
    return Ellipsoid(Einv=np.linalg.inv(E), c=c.value)


def maximum_inscribed_ellipsoid(vertices, rcond=None):
    """Compute the maximum inscribed ellipsoid for a polyhedron represented by
    a set of vertices.

    Returns the ellipsoid.
    """
    # rowspace
    R = orth(vertices.T, rcond=rcond)

    # project onto the rowspace
    # this allows us to handle degenerate sets of points that live in a
    # lower-dimensional subspace than R^d
    P = vertices @ R

    # solve the problem a possibly lower-dimensional space where the set of
    # vertices is full-rank
    A, b = polyhedron_span_to_face_form(P)
    ell = _maximum_inscribed_ellipsoid_inequality_form(A, b)

    # unproject back into the original space
    Einv = R @ ell.Einv @ R.T
    c = R @ ell.c
    return Ellipsoid(Einv=Einv, c=c)


def positive_definite_distance(A, B):
    """Geodesic distance between two positive definite matrices A and B.

    This metric is coordinate-frame invariant. See (Lee and Park, 2018) for
    more details.

    Parameters
    ----------
    A : np.ndarray
    B : np.ndarray

    Returns
    -------
    : float
        The non-negative geodesic between A and B.
    """
    C = np.linalg.solve(A, B)
    eigs = np.linalg.eigvals(C)
    # TODO: (Lee, Wensing, and Park, 2020) includes the factor of 0.5
    return np.sqrt(0.5 * np.sum(np.log(eigs) ** 2))
