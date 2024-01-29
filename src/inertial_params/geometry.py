"""Polyhedral and ellipsoidal geometry."""
from collections.abc import Iterable

import numpy as np
import cvxpy as cp
import cdd
from scipy.linalg import sqrtm, orth, null_space
from scipy.spatial import ConvexHull

from inertial_params.random import random_weight_vectors


class SpanForm:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)

    @classmethod
    def from_cdd_matrix(cls, mat):
        M = np.array([mat[i] for i in range(mat.row_size)])
        if mat.row_size == 0:
            return None

        # we are assuming a closed polyhedron, so there are no rays
        t = M[:, 0]
        assert np.allclose(t, 1.0)

        # vertices
        vertices = M[:, 1:]
        return cls(vertices)

    def to_face_form(self):
        # span form
        n = self.vertices.shape[0]
        Smat = cdd.Matrix(np.hstack((np.ones((n, 1)), self.vertices)))
        Smat.rep_type = cdd.RepType.GENERATOR

        # polyhedron
        poly = cdd.Polyhedron(Smat)

        # general face form is Ax <= b, which cdd stores as one matrix [b -A]
        Fmat = poly.get_inequalities()
        return FaceForm.from_cdd_matrix(Fmat)


class FaceForm:
    def __init__(self, A_ineq, b_ineq, A_eq=None, b_eq=None):
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.A_eq = A_eq
        self.b_eq = b_eq

    @classmethod
    def from_cdd_matrix(cls, mat):
        M = np.array([mat[i] for i in range(mat.row_size)])
        b = M[:, 0]
        A = -M[:, 1:]

        ineq_idx = np.array(
            [idx for idx in range(mat.row_size) if idx not in mat.lin_set]
        )
        eq_idx = np.array([idx for idx in mat.lin_set])

        return cls(
            A_ineq=A[ineq_idx, :],
            b_ineq=b[ineq_idx],
            A_eq=A[eq_idx, :] if len(eq_idx) > 0 else None,
            b_eq=b[eq_idx] if len(eq_idx) > 0 else None,
        )

    @property
    def spans_linear(self):
        return self.A_eq is not None

    def __repr__(self):
        return f"FaceForm(A_ineq={self.A_ineq}, A_eq={self.A_eq}, b_ineq={self.b_ineq}, b_eq={self.b_eq})"

    def stack(self, other):
        """Combine two face forms together."""
        A_ineq = np.vstack((self.A_ineq, other.A_ineq))
        b_ineq = np.concatenate((self.b_ineq, other.b_ineq))

        if self.spans_linear or other.spans_linear:
            A_eq = np.vstack([A for A in [self.A_eq, other.A_eq] if A is not None])
            b_eq = np.concatenate([b for b in [self.b_eq, other.b_eq] if b is not None])
        else:
            A_eq = None
            b_eq = None

        return FaceForm(A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq)

    def to_span_form(self):
        """Convert to span form (V-rep)."""
        A = np.vstack([A for A in [self.A_ineq, self.A_eq] if A is not None])
        b = np.concatenate([b for b in [self.b_ineq, self.b_eq] if b is not None])
        lin_set = frozenset(range(self.b_ineq.shape[0], b.shape[0]))

        S = np.hstack((b[:, None], -A))
        Smat = cdd.Matrix(S)
        Smat.lin_set = lin_set
        Smat.rep_type = cdd.RepType.INEQUALITY

        poly = cdd.Polyhedron(Smat)

        Fmat = poly.get_generators()
        return SpanForm.from_cdd_matrix(Fmat)


class ConvexPolyhedron:
    """A convex polyhedron in ``dim`` dimensions.

    The ``__init__`` method accepts either or both of the span (V-rep) and face
    (H-rep) forms of the polyhedron. If neither is provided, an error is
    raised. It is typically more convenient to construct the polyhedron using
    ``from_vertices`` or ``from_halfspaces``.

    Parameters
    ----------
    face_form : FaceForm or None
        The face form of the polyhedron.
    span_form : SpanForm or None
        The span form of the polyhedron.
    """

    def __init__(self, face_form=None, span_form=None):
        if face_form is None:
            face_form = span_form.to_face_form()
        if span_form is None:
            span_form = face_form.to_span_form()

        self.span_form = span_form
        self.face_form = face_form

    @classmethod
    def from_vertices(cls, vertices, prune=False):
        """Construct the polyhedron from a set of vertices.

        Parameters
        ----------
        vertices : np.ndarray, shape (nv, dim)
            The extremal points of the polyhedron.
        prune : bool
            If ``True``, the vertices will be pruned to eliminate any non-extremal
            points.
        """
        if prune:
            vertices = convex_hull(vertices)
        span_form = SpanForm(vertices)
        return cls(span_form=span_form)

    @classmethod
    def from_halfspaces(cls, A, b):
        """Construct the polyhedron from a set of halfspaces.

        The polyhedron is the set {x | Ax <= b}. For degenerate cases with
        linear *equality* constraints, use the ``__init__`` method to pass a
        face form directly.

        Parameters
        ----------
        A : np.ndarray
            The matrix of halfspace normals.
        b : np.ndarray
            The vector of halfspace offsets.
        """
        face_form = FaceForm(A_ineq=A, b_ineq=b)
        return cls(face_form=face_form)

    def __repr__(self):
        return f"ConvexPolyhedron(vertices={self.vertices})"

    @property
    def vertices(self):
        """The extremal points of the polyhedron."""
        return self.span_form.vertices

    @property
    def nv(self):
        """The number of vertices."""
        return self.vertices.shape[0]

    @property
    def dim(self):
        """The dimension of the ambient space."""
        return self.vertices.shape[1]

    # def is_box(self):
    #     if self.A.shape
    #     pass

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
        n = points.shape[0]

        B_ineq = np.tile(self.face_form.b_ineq, (n, 1)).T
        ineq = self.face_form.A_ineq @ points.T <= B_ineq + tol

        # degenerate polyhedra may contain equality constraints
        if self.face_form.spans_linear:
            B_eq = np.tile(self.face_form.b_eq, (n, 1)).T
            eq = np.abs(self.face_form.A_eq @ points.T - B_eq) <= tol
            result = np.logical_and(eq.all(axis=0), ineq.all(axis=0))
        else:
            result = ineq.all(axis=0)

        # convert to scalar if only one point was tested
        if result.size == 1:
            return result.item()
        return result

    def contains_polyhedron(self, other, tol=1e-8):
        """Test if this polyhedron contains another one.

        Parameters
        ----------
        other : ConvexPolyhedron
            The other polyhedron to check.
        tol : float, non-negative
            The numerical tolerance for membership.

        Returns
        -------
        : bool
            ``True`` if this polyhedron contains the other, ``False`` otherwise.
        """
        return self.contains(other.vertices, tol=tol).all()

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

    def minimum_bounding_ellipsoid(self, rcond=None):
        """Construct the minimum-volume bounding ellipsoid for this polyhedron."""
        return minimum_bounding_ellipsoid(self.vertices, rcond=rcond)

    def maximum_inscribed_ellipsoid(self, rcond=None):
        """Construct the maximum-volume ellipsoid inscribed in this polyhedron."""
        return maximum_inscribed_ellipsoid(self.vertices, rcond=rcond)

    def intersect(self, other):
        """Intersect this polyhedron with another one.

        Parameters
        ----------
        other : ConvexPolyhedron
            The other polyhedron.

        Returns
        -------
        : ConvexPolyhedron or None
            The intersection, which is another ``ConvexPolyhedron``, or
            ``None`` if the two polyhedra do not intersect.
        """
        assert isinstance(other, ConvexPolyhedron)

        span_form = self.face_form.stack(other.face_form).to_span_form()
        if span_form is None:
            return None
        return ConvexPolyhedron(span_form=span_form)


def _box_vertices(half_extents, center, rotation):
    """Generate the vertices of an oriented box."""
    x, y, z = half_extents
    v = np.array(
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
    return (rotation @ v.T).T + center


class AxisAlignedBox(ConvexPolyhedron):
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

    def __init__(self, half_extents, center=None, rotation=None):
        self.half_extents = np.array(half_extents)
        assert self.half_extents.shape == (3,)
        assert np.all(self.half_extents >= 0)

        if center is None:
            center = np.zeros(3)
        self.center = np.array(center)

        if rotation is None:
            rotation = np.eye(3)
        self.rotation = np.array(rotation)
        assert self.rotation.shape == (3, 3)

        vertices = _box_vertices(self.half_extents, self.center, self.rotation)
        super().__init__(span_form=SpanForm(vertices))

    @classmethod
    def cube(cls, half_extent, center=None, rotation=None):
        """Construct a cube."""
        assert half_extent >= 0
        half_extents = half_extent * np.ones(3)
        return cls(half_extents, center=center, rotation=rotation)

    @classmethod
    def from_side_lengths(cls, side_lengths, center=None, rotation=None):
        """Construct a box with given side lengths."""
        return cls(0.5 * side_lengths, center=center, rotation=rotation)

    @classmethod
    def from_two_vertices(cls, v1, v2):
        """Construct an axis-aligned box from two opposed vertices."""
        center = 0.5 * (v1 + v2)
        half_extents = 0.5 * (np.maximum(v1, v2) - np.minimum(v1, v2))
        return cls(half_extents, center=center)

    @classmethod
    def from_points_to_bound(cls, points):
        """Construct the smallest axis-aligned box that contains all of the points."""
        v_min = np.min(points, axis=0)
        v_max = np.max(points, axis=0)
        return cls.from_two_vertices(v_min, v_max)

    def __repr__(self):
        return f"Box(half_extents={self.half_extents}, center={self.center}, rotation={self.rotation})"

    @property
    def side_lengths(self):
        """The side lengths of the box."""
        return 2 * self.half_extents

    @property
    def volume(self):
        """The volume of the box."""
        return np.product(self.side_lengths)

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
        L = -self.half_extents
        U = self.half_extents

        x = np.linspace(L[0], U[0], n)
        y = np.linspace(L[1], U[1], n)
        z = np.linspace(L[2], U[2], n)

        # TODO this is inefficient
        points = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    point = self.rotation @ [x[i], y[j], z[k]] + self.center
                    points.append(point)
        return np.array(points)

    def as_ellipsoidal_intersection(self):
        """Construct a set of ellipsoids, the intersection of which is the box."""
        r1, r2, r3 = self.half_extents
        A1 = self.rotation @ np.diag([1.0 / r1**2, 0, 0]) @ self.rotation.T
        A2 = self.rotation @ np.diag([0, 1.0 / r2**2, 0]) @ self.rotation.T
        A3 = self.rotation @ np.diag([0, 0, 1.0 / r3**2]) @ self.rotation.T
        return [Ellipsoid(A, self.center) for A in [A1, A2, A3]]

    def transform(self, translation=None, rotation=None):
        """Apply a rigid transformation to the box."""
        center = self.center.copy()
        orn = self.rotation.copy()

        if rotation is not None:
            center = rotation @ center
            orn = rotation @ orn
        if translation is not None:
            center += translation
        return AxisAlignedBox(
            half_extents=self.half_extents.copy(), center=center, rotation=orn
        )

    def rotate_about_center(self, rotation):
        """Rotate the box about its center point.

        Parameters
        ----------
        rotation : np.ndarray, shape (3, 3)
            Rotation matrix.

        Returns
        -------
        : AxisAlignedBox
            A new box that has been rotated about its center point.
        """
        rotation = rotation @ self.rotation
        center = self.center.copy()
        half_extents = self.half_extents.copy()
        return AxisAlignedBox(
            half_extents=half_extents, center=center, rotation=rotation
        )


class Cylinder:
    """A cylinder in three dimensions.

    Parameters
    ----------
    length : float, positive
        The length along the longitudinal axis.
    radius : float, positive
        The radius of the cross-section.
    axis : np.ndarray, shape (3,)
        The main axis. If not provided, defaults to ``[0, 0, 1]``. The axis is
        normalized to unit length.
    center : np.ndarray, shape (3,)
        The center of the cylinder. If not provided, defaults to the origin.
    """

    def __init__(self, length, radius, axis=None, center=None):
        assert length > 0
        assert radius > 0

        if axis is None:
            axis = np.array([0, 0, 1])
        else:
            axis = np.array(axis) / np.linalg.norm(axis)
        self.axis = axis
        self.length = length
        self.radius = radius

        if center is None:
            center = np.zeros_like(axis)
        self.center = np.array(center)

        self.U = null_space(self.axis[None, :])

    def as_ellipsoidal_intersection(self):
        """Construct a set of ellipsoids, the intersection of which is the cylinder."""
        # TODO probably just make these properties of the cylinder
        A1 = 4 * np.outer(self.axis, self.axis) / self.length**2
        A2 = self.U @ self.U.T / self.radius**2
        return [Ellipsoid(A, self.center) for A in [A1, A2]]

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
        points = np.atleast_2d(points)
        xs = points - self.center
        contained_lengthwise = np.abs(xs @ self.axis) <= self.length / 2
        contained_transverse = np.array(
            [x @ self.U @ self.U.T @ x <= self.radius**2 for x in xs], dtype=bool
        )
        assert contained_lengthwise.shape == contained_transverse.shape
        contained = np.logical_and(contained_lengthwise, contained_transverse)
        return np.squeeze(contained)

    def inscribed_box(self):
        # TODO once box has rotation information
        pass

    def bounding_box(self):
        pass


# TODO rename c -> center, possibly rename Einv also
class Ellipsoid:
    """Ellipsoid with a variety of representations."""

    def __init__(self, Einv, center=None):
        assert Einv.ndim == 2
        assert Einv.shape[0] == Einv.shape[1]
        self.Einv = Einv
        self.dim = self.Einv.shape[0]

        if center is None:
            center = np.zeros(self.dim)
        self.center = np.array(center)

        self.rank = np.linalg.matrix_rank(self.Einv)

    def __repr__(self):
        return f"Ellipsoid(center={self.center}, Einv={self.Einv})"

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
        return cls(Einv=Einv, center=center)

    @classmethod
    def from_half_extents(cls, half_extents, center=None):
        """Construct an ellipsoid from its half extents, which are the lengths
        of the semi-major axes.

        Parameters
        ----------
        half_extents : np.ndarray, shape (d,)
            The lengths of the semi-major axes, where ``d`` is the dimension.
        center : np.ndarray, shape (d,)
            Optional center point of the ellipsoid.
        """
        half_extents = np.array(half_extents)
        Einv = np.diag(1.0 / half_extents**2)
        return cls(Einv=Einv, center=center)

    @classmethod
    def from_Ab(cls, A, b, rcond=None):
        Einv = A @ A

        # use least squares instead of direct solve in case we have a
        # degenerate ellipsoid
        center = np.linalg.lstsq(A, -b, rcond=rcond)[0]
        return cls(Einv=Einv, center=center)

    @classmethod
    def from_Q(cls, Q):
        assert Q.shape[0] == Q.shape[1]
        dim = Q.shape[0] - 1
        Einv = -Q[:dim, :dim]
        q = Q[:dim, dim]
        center = np.linalg.solve(Einv, q)
        return cls(Einv=Einv, center=center)

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
        return -self.A @ self.center

    @property
    def Q(self):
        """Q representation of the ellipsoid.

        .. math::
           \\mathcal{E} = \\{x\\in\\mathbb{R}^d \\mid \\tilde{x}^TQ\\tilde{q}\\geq 0\\}
        """

        Q = np.zeros((self.dim + 1, self.dim + 1))
        Q[: self.dim, : self.dim] = -self.Einv
        Q[: self.dim, self.dim] = self.Einv @ self.center
        Q[self.dim, : self.dim] = Q[: self.dim, self.dim]
        Q[self.dim, self.dim] = 1 - self.center @ self.Einv @ self.center
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
            p = points - self.center
            return p @ self.Einv @ p <= 1 + tol
        elif points.ndim == 2:
            ps = points - self.center
            return np.array([p @ self.Einv @ p <= 1 + tol for p in ps])
        else:
            raise ValueError(
                f"points must have 1 or 2 dimensions, but has {points.ndim}."
            )

    def transform(self, rotation=None, translation=None):
        """Apply an affine transform to the ellipsoid.

        Parameters
        ----------
        rotation : np.ndarray, shape (d, d)
            Rotation matrix.
        translation : np.ndarray, shape (d,)
            Translation vector.

        Returns
        -------
        : Ellipsoid
            A new ellipsoid that has been rigidly transformed.
        """
        if rotation is None and translation is None:
            return Ellipsoid(Einv=self.Einv.copy(), center=self.center.copy())

        if rotation is None:
            dim = translation.shape[0]
            rotation = np.eye(dim)
        if translation is None:
            dim = rotation.shape[0]
            translation = np.zeros(dim)

        Einv = rotation @ self.Einv @ rotation.T
        center = rotation @ self.center + translation
        return Ellipsoid(Einv=Einv, center=center)

    def rotate_about_center(self, rotation):
        """Rotate the ellipsoid about its center point.

        Parameters
        ----------
        rotation : np.ndarray, shape (d, d)
            Rotation matrix.

        Returns
        -------
        : Ellipsoid
            A new ellipsoid that has been rigidly transformed.
        """
        Einv = rotation @ self.Einv @ rotation.T
        return Ellipsoid(Einv=Einv, center=self.center.copy())


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
    return Ellipsoid(Einv=np.linalg.inv(E), center=c.value)


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
    face_form = SpanForm(P).to_face_form()
    assert not face_form.spans_linear, "Vertices have a linear constraint!"
    ell = _maximum_inscribed_ellipsoid_inequality_form(
        face_form.A_ineq, face_form.b_ineq
    )

    # unproject back into the original space
    Einv = R @ ell.Einv @ R.T
    center = R @ ell.center
    return Ellipsoid(Einv=Einv, center=center)


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
