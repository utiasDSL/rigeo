from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import cvxpy as cp
from scipy.linalg import orth, sqrtm, null_space

from ..polydd import SpanForm, FaceForm
from ..util import clean_transform
from ..constraint import schur, pim_must_equal_param_var
from ..random import (
    random_weight_vectors,
    random_points_in_ball,
    random_points_on_hypersphere,
)
from ..inertial import InertialParameters
from .base import Shape


# set to True to use alternative trace constraints for degenerate ellipsoids
USE_ELLIPSOID_TRACE_REALIZABILITY_CONSTRAINTS = False


def _inv_with_zeros(a, tol=1e-8):
    """Invert an array that may contain zeros.

    The inverse of an entry that is (close to) zero is replaced with np.inf.
    """
    zero_mask = np.isclose(a, 0, rtol=0, atol=tol)
    out = np.inf * np.ones_like(a)
    np.divide(1.0, a, out=out, where=~zero_mask)
    return out


def _Q_matrix(S, c, bound=1):
    """Helper to build an ellipsoid's Q matrix.

    Parameters
    ----------
    S : np.ndarray, shape (dim, dim)
        The ellipsoid shape matrix.
    c : np.ndarray, shape (dim,)
        The ellipsoid center.
    bound : float
        The bound for points to be included in the ellipsoid: ``(p - c) @ S @
        (p - c) <= bound``. Typically this is one, but could also be zero if the
        ellipsoid has no length along some direction.

    Returns
    -------
    : np.ndarray, shape (dim + 1, dim + 1)
        The Q matrix.
    """
    dim = S.shape[0]
    Q = np.zeros((dim + 1, dim + 1))
    Q[:dim, :dim] = -S
    Q[:dim, dim] = S @ c
    Q[dim, :dim] = Q[:dim, dim]
    Q[dim, dim] = bound - c @ S @ c
    return Q


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


def _mie_inequality_form(A, b, sphere=False, solver=None):
    """Compute the maximum inscribed ellipsoid for an inequality-form
    polyhedron P = {x | Ax <= b}.

    See :cite:t:`boyd2004convex`, Section 8.4.1.

    Returns the ellipsoid.
    """

    dim = A.shape[1]
    n = b.shape[0]

    B = cp.Variable((dim, dim), PSD=True)
    c = cp.Variable(dim)

    objective = cp.Maximize(cp.log_det(B))
    constraints = [
        cp.norm2(B @ A[i, :]) + A[i, :] @ c <= b[i] for i in range(n)
    ]
    if sphere:
        # if we want a sphere, then A is a multiple of the identity matrix
        r = cp.Variable(1)
        constraints.append(B == r * np.eye(dim))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    E = B.value @ B.value
    return Ellipsoid.from_shape_matrix(S=np.linalg.inv(E), center=c.value)


class ConvexPolyhedron(Shape):
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

        if not span_form.bounded():
            raise ValueError("Only bounded polyhedra are supported.")

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
            If ``True``, the vertices will be pruned to eliminate any
            non-extremal points.
        """
        span_form = SpanForm(vertices=vertices)
        if prune:
            span_form = span_form.canonical()
        return cls(span_form=span_form)

    @classmethod
    def from_halfspaces(cls, A, b, prune=False):
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
        prune : bool
            If ``True``, the halfspaces will be pruned to eliminate any
            redundancies.
        """
        face_form = FaceForm(A_ineq=A, b_ineq=b)
        if prune:
            face_form = face_form.canonical()
        return cls(face_form=face_form)

    def __repr__(self):
        return f"ConvexPolyhedron(vertices={self.vertices})"

    @property
    def A(self):
        """Matrix part of the face form (normals)."""
        return self.face_form.A

    @property
    def b(self):
        """Vector part of the face form (offsets)."""
        return self.face_form.b

    @property
    def nf(self):
        """Number of faces."""
        return self.face_form.nf

    @property
    def vertices(self):
        """The extremal points of the polyhedron."""
        return self.span_form.vertices

    @property
    def nv(self):
        """Number of vertices."""
        return self.span_form.nv

    @property
    def dim(self):
        """The dimension of the ambient space."""
        return self.span_form.dim

    def contains(self, points, tol=1e-8):
        points = np.array(points)
        if points.ndim == 1:
            return np.all(self.A @ points <= self.b + tol)
        return np.array([np.all(self.A @ p <= self.b + tol) for p in points])

    def must_contain(self, points, scale=1.0):
        if points.ndim == 1:
            points = [points]
        return [self.A @ p <= scale * self.b for p in points]

    def is_same(self, other, tol=1e-8):
        if not isinstance(other, self.__class__):
            return False
        return self.contains_polyhedron(
            other, tol=tol
        ) and other.contains_polyhedron(self, tol=tol)

    def random_points(self, shape=1, rng=None):
        # NOTE: this is not uniform sampling!
        if np.isscalar(shape):
            shape = (shape,)
        full_shape = tuple(shape) + (self.nv,)
        w = random_weight_vectors(full_shape, rng=rng)
        points = w @ self.vertices
        if shape == (1,):
            return np.squeeze(points)
        return points

    def aabb(self):
        return Box.from_points_to_bound(self.vertices)

    def mbe(self, rcond=None, sphere=False, solver=None):
        """Construct the minimum-volume bounding ellipsoid for this polyhedron."""
        return mbe_of_points(
            self.vertices, rcond=rcond, sphere=sphere, solver=solver
        )

    # def mie(self, rcond=None, sphere=False, solver=None):
    #     """Construct the maximum-volume ellipsoid inscribed in this polyhedron."""
    #     return mie(self.vertices, rcond=rcond, sphere=sphere, solver=solver)

    def mie(self, rcond=None, sphere=False, solver=None):
        """Compute the maximum inscribed ellipsoid for a polyhedron represented by
        a set of vertices.

        Returns the ellipsoid.
        """
        # rowspace
        r = self.vertices[0]
        R = orth((self.vertices - r).T, rcond=rcond)
        rank = R.shape[1]

        # project onto the rowspace
        # this allows us to handle degenerate sets of points that live in a
        # lower-dimensional subspace than R^d
        P = (self.vertices - r) @ R

        # solve the problem a possibly lower-dimensional space where the set of
        # vertices is full-rank
        face_form = SpanForm(P).to_face_form()
        ell = _mie_inequality_form(
            face_form.A, face_form.b, sphere=sphere, solver=solver
        )

        # unproject
        half_extents = np.zeros(self.vertices.shape[1])
        half_extents[:rank] = ell.half_extents

        N = null_space(R.T, rcond=rcond)
        rotation = np.hstack((R @ ell.rotation, N))

        center = R @ ell.center + r
        return Ellipsoid(
            half_extents=half_extents, rotation=rotation, center=center
        )

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

    def _can_realize_tetrahedron(self, params, tol=0):
        # center of mass must be inside the shape
        if not self.contains(params.com, tol=tol):
            return False

        c = np.append(params.com, 1)
        V = np.vstack((self.vertices.T, np.ones(4)))
        ms = np.linalg.solve(V, params.mass * c)

        Vs = np.array([np.outer(v, v) for v in self.vertices])
        H_max = sum([m * V for m, V in zip(ms, Vs)])

        return np.min(np.linalg.eigvals(H_max - params.H)) >= -tol

    def can_realize(self, params, eps=0, **kwargs):
        assert (
            self.dim == 3
        ), "Shape must be 3-dimensional to realize inertial parameters."
        # assert tol >= 0, "Numerical tolerance cannot be negative."
        if not params.consistent(eps=eps):
            return False

        # special case for tetrahedra: this does not require solving an
        # optimization problem
        if self.nv == 4:
            return self._can_realize_tetrahedron(params, tol=0)  # TODO fix tol

        J = cp.Variable((4, 4), PSD=True)

        objective = cp.Minimize([0])  # feasibility problem
        constraints = self.must_realize(J) + [J == params.J]
        problem = cp.Problem(objective, constraints)
        problem.solve(**kwargs)
        return problem.status == "optimal"

    def must_realize(self, param_var, eps=0):
        assert (
            self.dim == 3
        ), "Shape must be 3-dimensional to realize inertial parameters."
        J, psd_constraints = pim_must_equal_param_var(param_var, eps)
        m = J[3, 3]
        h = J[:3, 3]
        H = J[:3, :3]

        Vs = np.array([np.outer(v, v) for v in self.vertices])
        ms = cp.Variable(self.nv)

        return psd_constraints + [
            ms >= 0,
            m == cp.sum(ms),
            h == ms.T @ self.vertices,
            H << cp.sum([μ * V for μ, V in zip(ms, Vs)]),
        ]

    def transform(self, rotation=None, translation=None):
        rotation, translation = clean_transform(
            rotation=rotation, translation=translation, dim=self.dim
        )
        new_vertices = self.vertices @ rotation.T + translation
        return self.from_vertices(new_vertices)

    def vertex_point_mass_params(self, mass):
        """Compute the inertial parameters corresponding to a system of point
        masses located at the vertices.

        Parameters
        ----------
        mass : float or np.ndarray, shape (self.nv,)
            A single scalar represents the total mass which is uniformly
            distributed among the vertices. Otherwise represents the mass for
            each individual vertex.

        Returns
        -------
        : InertialParameters
            The parameters representing the point mass system.
        """
        if np.isscalar(mass):
            masses = mass * np.ones(self.nv) / self.nv
        else:
            masses = np.array(mass)

        assert masses.shape == (self.nv,)
        assert np.all(masses >= 0)

        return InertialParameters.from_point_masses(
            masses=masses, points=self.vertices
        )


class Box(ConvexPolyhedron):
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
    rotation : np.ndarray, shape (3, 3)
        The orientation of the box. If ``rotation=np.eye(3)``, then the box is
        axis-aligned.
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
        return cls(
            0.5 * np.array(side_lengths), center=center, rotation=rotation
        )

    @classmethod
    def from_two_vertices(cls, v1, v2):
        """Construct an axis-aligned box from two opposed vertices."""
        v1 = np.array(v1)
        v2 = np.array(v2)
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
        return np.prod(self.side_lengths)

    @property
    def diaglen(self):
        """Length of the box's diagonal."""
        return 2 * np.linalg.norm(self.half_extents)

    def random_points(self, shape=1, rng=None):
        """Uniformly sample points contained in the box."""
        if np.isscalar(shape):
            shape = (shape,)
        n = np.prod(shape)

        rng = np.random.default_rng(rng)
        points = rng.uniform(low=-1, high=1, size=(n, 3)) * self.half_extents
        points = (self.rotation @ points.T).T + self.center
        if shape == (1,):
            return np.squeeze(points)
        return points.reshape(shape + (3,))

    def random_points_on_surface(self, shape=1, rng=None):
        """Uniformly sample points on the surface of the box."""
        if np.isscalar(shape):
            shape = (shape,)
        n = np.prod(shape)

        # generate some random points inside the box
        # TODO would probably be nicer to sample the number per face first,
        # like in the edges function below
        rng = np.random.default_rng(rng)
        points = self.random_points(n, rng=rng)

        x, y, z = self.half_extents
        Ax = 4 * y * z
        Ay = 4 * x * z
        Az = 4 * x * y
        A = 2 * (Ax + Ay + Az)
        pvals = np.array([Ax, Ax, Ay, Ay, Az, Az]) / A

        # randomly project each point on one of the sides proportional to its
        # area
        counts = rng.multinomial(n=n, pvals=pvals)
        idx = np.cumsum(counts)
        points[0 : idx[0], 0] = x
        points[idx[0] : idx[1], 0] = -x
        points[idx[1] : idx[2], 1] = y
        points[idx[2] : idx[3], 1] = -y
        points[idx[3] : idx[4], 2] = z
        points[idx[4] :, 2] = -z

        # shuffle to regain randomness w.r.t. the sides
        rng.shuffle(points)

        points = (self.rotation @ points.T).T + self.center
        if shape == (1,):
            return np.squeeze(points)
        return points.reshape(shape + (3,))

    def random_points_on_edges(self, shape=1, rng=None):
        """Uniformly sample points on the edges of the box."""
        if np.isscalar(shape):
            shape = (shape,)
        n = np.prod(shape)

        rng = np.random.default_rng(rng)

        # draw samples from each edge proportional to its length
        x, y, z = self.half_extents
        L = 4 * (x + y + z)
        pvals = np.array([x, x, x, x, y, y, y, y, z, z, z, z]) / L
        counts = np.concatenate(([0], rng.multinomial(n=n, pvals=pvals)))
        idx = np.cumsum(counts)

        # randomly sample on each edge
        points = np.zeros((n, 3))

        # edges along x-direction
        points[: idx[4], 0] = x * rng.uniform(low=-1, high=1, size=idx[4])
        for i, yz in enumerate([[y, z], [y, -z], [-y, -z], [-y, z]]):
            points[idx[i] : idx[i + 1], 1:] = yz

        # edges along y-direction
        points[idx[4] : idx[8], 1] = y * rng.uniform(
            low=-1, high=1, size=idx[8] - idx[4]
        )
        for i, xz in enumerate([[x, z], [x, -z], [-x, -z], [-x, z]]):
            points[idx[4 + i] : idx[4 + i + 1], [0, 2]] = xz

        # edges along z-direction
        points[idx[8] :, 2] = z * rng.uniform(
            low=-1, high=1, size=idx[12] - idx[8]
        )
        for i, xy in enumerate([[x, y], [x, -y], [-x, -y], [-x, y]]):
            points[idx[8 + i] : idx[8 + i + 1], :2] = xy

        rng.shuffle(points)

        points = (self.rotation @ points.T).T + self.center
        if shape == (1,):
            return np.squeeze(points)
        return points.reshape(shape + (3,))

    def on_surface(self, points, tol=1e-8):
        points = np.atleast_2d(points)
        contained = self.contains(points, tol=tol)

        # for each point, check if at least one coordinate is on a face
        # TODO use tol
        x, y, z = self.half_extents
        points = (points - self.center) @ self.rotation
        x_mask = np.isclose(np.abs(points[:, 0]), x)
        y_mask = np.isclose(np.abs(points[:, 1]), y)
        z_mask = np.isclose(np.abs(points[:, 2]), z)
        return contained & (x_mask | y_mask | z_mask)

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

    def transform(self, rotation=None, translation=None):
        rotation, translation = clean_transform(
            rotation=rotation, translation=translation, dim=3
        )
        new_rotation = rotation @ self.rotation
        new_center = rotation @ self.center + translation
        half_extents = self.half_extents.copy()
        return Box(
            half_extents=half_extents, center=new_center, rotation=new_rotation
        )

    def rotate_about_center(self, rotation):
        """Rotate the box about its center point.

        Parameters
        ----------
        rotation : np.ndarray, shape (3, 3)
            Rotation matrix.

        Returns
        -------
        : Box
            A new box that has been rotated about its center point.
        """
        assert rotation.shape == (3, 3)
        new_rotation = rotation @ self.rotation
        center = self.center.copy()
        half_extents = self.half_extents.copy()
        return Box(
            half_extents=half_extents, center=center, rotation=new_rotation
        )

    def mie(self, rcond=None, sphere=False, solver=None):
        if sphere:
            radius = np.min(self.half_extents)
            return Ellipsoid.sphere(radius=radius, center=self.center)
        return Ellipsoid(
            half_extents=self.half_extents,
            center=self.center,
            rotation=self.rotation,
        )

    def mbe(self, rcond=None, sphere=False, solver=None):
        half_extents = self.half_extents * np.sqrt(3)  # TODO check this
        if sphere:
            radius = np.max(half_extents)
            return Ellipsoid.sphere(radius=radius, center=self.center)
        return Ellipsoid(
            half_extents=half_extents,
            center=self.center,
            rotation=self.rotation,
        )

    def as_poly(self):
        """Convert the box to a general convex polyhedron.

        Returns
        -------
        : ConvexPolyhedron
        """
        return ConvexPolyhedron.from_vertices(self.vertices)

    def uniform_density_params(self, mass):
        """Generate the inertial parameters corresponding to a uniform mass density.

        The inertial parameters are generated with respect to the origin.

        Parameters
        ----------
        mass : float, non-negative
            The mass of the body.

        Returns
        -------
        : InertialParameters
            The inertial parameters.
        """
        assert mass >= 0, "Mass must be non-negative."

        H = mass * np.diag(self.half_extents**2) / 3.0
        return InertialParameters(mass=mass, h=np.zeros(3), H=H).transform(
            rotation=self.rotation, translation=self.center
        )

    def hollow_density_params(self, mass):
        """Generate the inertial parameters corresponding to a hollow box.

        In other words, all of the mass is uniformly distributed on the surface
        of the box. The inertial parameters are generated with respect to
        the origin.

        Parameters
        ----------
        mass : float, non-negative
            The mass of the body.

        Returns
        -------
        : InertialParameters
            The inertial parameters.
        """
        assert mass >= 0, "Mass must be non-negative."

        x, y, z = self.half_extents
        d = 3 * (x * y + x * z + y * z)
        Hxx = x**2 * (x * y + x * z + 3 * y * z) / d
        Hyy = y**2 * (x * y + 3 * x * z + y * z) / d
        Hzz = z**2 * (3 * x * y + x * z + y * z) / d
        H = np.diag([Hxx, Hyy, Hzz])
        return InertialParameters(mass=mass, h=np.zeros(3), H=H).transform(
            rotation=self.rotation, translation=self.center
        )

    def wireframe_density_params(self, mass):
        assert mass >= 0, "Mass must be non-negative."

        x, y, z = self.half_extents
        d = x + y + z
        Hxx = x**2 * (x / 3 + y + z) / d
        Hyy = y**2 * (x + y / 3 + z) / d
        Hzz = z**2 * (x + y + z / 3) / d
        H = np.diag([Hxx, Hyy, Hzz])
        return InertialParameters(mass=mass, h=np.zeros(3), H=H).transform(
            rotation=self.rotation, translation=self.center
        )


class Ellipsoid(Shape):
    """Ellipsoid in ``dim`` dimensions.

    The ellipsoid may be degenerate, which means that one or more of the half
    extents is zero and it has no volume.
    """

    # The ellipsoid may be degenerate in two ways:
    # 1. If one or more half extents is infinite, then the ellipsoid is unbounded
    #    along one or more axes.
    # 2. If one or more half extents is zero, then the ellipsoid actually lives
    #    in a lower-dimensional subspace.
    def __init__(self, half_extents, rotation=None, center=None):
        if np.isscalar(half_extents):
            half_extents = [half_extents]
        self.half_extents = np.array(half_extents)

        assert np.all(
            self.half_extents >= 0
        ), "Half extents cannot be negative."
        # assert np.all(np.isfinite(self.half_extents)), "Half extents must be finite."

        self.half_extents_inv = _inv_with_zeros(self.half_extents)

        if rotation is None:
            rotation = np.eye(self.dim)
        self.rotation = np.array(rotation)
        assert self.rotation.shape == (self.dim, self.dim)

        if center is None:
            center = np.zeros(self.dim)
        elif np.isscalar(center):
            center = [center]
        self.center = np.array(center)
        assert self.center.shape == (self.dim,)

    @property
    def dim(self):
        return self.half_extents.shape[0]

    @property
    def S(self):
        """Shape matrix."""
        return (
            self.rotation
            @ np.diag(self.half_extents_inv**2)
            @ self.rotation.T
        )

    @property
    def E(self):
        """Inverse of the shape matrix."""
        return self.rotation @ np.diag(self.half_extents**2) @ self.rotation.T

    @property
    def rank(self):
        return np.count_nonzero(self.half_extents)

    @property
    def volume(self):
        """The volume of the ellipsoid."""
        return 4 * np.pi * np.prod(self.half_extents) / 3

    def __repr__(self):
        return f"Ellipsoid(half_extents={self.half_extents}, center={self.center}, rotation={self.rotation})"

    @classmethod
    def sphere(cls, radius, center=None):
        """Construct a sphere.

        Parameters
        ----------
        radius : float
            Radius of the sphere.
        center : np.ndarray, shape (dim,)
            Optional center point of the sphere.
        """
        if center is None:
            center = np.zeros(3)
        else:
            center = np.array(center)

        dim = center.shape[0]
        half_extents = radius * np.ones(dim)
        return cls(half_extents=half_extents, center=center)

    @classmethod
    def from_shape_matrix(cls, S, center=None):
        # we can use eigh since S is symmetric
        eigs, rotation = np.linalg.eigh(S)

        half_extents_inv = np.sqrt(eigs)
        half_extents = _inv_with_zeros(half_extents_inv)

        return cls(half_extents=half_extents, rotation=rotation, center=center)

    @classmethod
    def from_affine(cls, A, b, rcond=None):
        """Construct an ellipsoid from an affine transformation of the unit ball."""
        S = A @ A

        # use least squares instead of direct solve in case we have a
        # degenerate ellipsoid
        center = np.linalg.lstsq(A, -b, rcond=rcond)[0]
        return cls.from_shape_matrix(S=S, center=center)

    def lower(self):
        """Project onto a ``rank``-dimensional subspace."""
        # TODO this needs to be finished and tested
        # TODO do I even want this?
        if rank == dim:
            return self
        nz = np.nonzero(self.half_extents)
        half_extents = self.half_extents[nz]
        U = self.rotation[:, nz]
        center = U.T @ self.center

    @property
    def A(self):
        """The matrix :math:`\\boldsymbol{A}` from when the ellipsoid is
        represented as an affine transformation of the unit ball.

        .. math::
           \\mathcal{E} = \\{x\\in\\mathbb{R}^d \\mid \\|Ax+b\\|^2\\leq 1\\}
        """
        # TODO ensure this is tested properly
        return self.rotation @ np.diag(self.half_extents_inv) @ self.rotation.T

    @property
    def b(self):
        """The vector :math:`\\boldsymbol{b}` from when the ellipsoid is
        represented as an affine transformation of the unit ball.

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
        return _Q_matrix(self.S, self.center)

    def _Q_degenerate(self):
        zero_mask = np.isclose(self.half_extents, 0)

        # non-zero directions treated normally
        # TODO is this correct?
        nonzero_axes = (
            self.half_extents_inv[~zero_mask] * self.rotation[:, ~zero_mask]
        )
        nonzero_Q = _Q_matrix(nonzero_axes @ nonzero_axes.T, self.center)

        # zero directions must have no spread in the mass distribution
        zero_axes = self.rotation[:, zero_mask]
        zero_Q = _Q_matrix(zero_axes @ zero_axes.T, self.center, bound=0)

        return nonzero_Q, zero_Q

    def _QV_degenerate(self):
        zero_mask = np.isclose(self.half_extents, 0)

        # Q matrix in the lower-dimensional space
        Sr = np.diag(self.half_extents_inv[~zero_mask] ** 2)
        Qr = _Q_matrix(S=Sr, c=np.zeros(self.rank))

        # V matrix projects lower-dimensional J up to 3D space
        V = np.zeros((4, self.rank + 1))
        V[:3, :-1] = self.rotation[:, ~zero_mask]
        V[:3, -1] = self.center
        V[3, -1] = 1

        return Qr, V

    def is_degenerate(self):
        """Check if the ellipsoid is degenerate.

        This means that it has zero volume, and lives in a lower dimension than
        the ambient one.

        Returns
        -------
        : bool
            Returns ``True`` if the ellipsoid is degenerate, ``False`` otherwise.
        """
        return self.rank < self.dim

    def is_infinite(self):
        return np.any(np.isinf(self.half_extents))

    def is_same(self, other, tol=1e-8):
        """Check if this ellipsoid is the same as another."""
        if not isinstance(other, self.__class__):
            return False
        return np.allclose(self.Q, other.Q, atol=tol)

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
        zero_mask = np.isclose(self.half_extents, 0)
        S_diag = np.diag(self.half_extents_inv[~zero_mask] ** 2)

        if points.ndim == 1:
            p = self.rotation.T @ (points - self.center)

            # value along degenerate dimension must be zero
            if not np.allclose(p[zero_mask], 0, rtol=0, atol=tol):
                return False

            return p[~zero_mask] @ S_diag @ p[~zero_mask] <= 1 + tol
        elif points.ndim == 2:
            ps = (points - self.center) @ self.rotation

            # degenerate dimensions
            res1 = np.all(
                np.isclose(ps[:, zero_mask], 0, rtol=0, atol=tol), axis=1
            )

            # nondegenerate dimensions
            res2 = np.array(
                [p[~zero_mask] @ S_diag @ p[~zero_mask] <= 1 + tol for p in ps]
            )

            # combine them
            return np.logical_and(res1, res2)
        else:
            raise ValueError(
                f"points must have 1 or 2 dimensions, but has {points.ndim}."
            )

    def must_contain(self, points, scale=1.0):
        inf_mask = np.isinf(self.half_extents)
        E_diag = np.diag(self.half_extents[~inf_mask] ** 2)

        if points.ndim == 1:
            points = [points]

        constraints = []
        for point in points:
            p = self.rotation.T @ (point - scale * self.center)
            c = schur(scale * E_diag, p[~inf_mask], scale) >> 0
            constraints.append(c)
        return constraints

    def can_realize(self, params, eps=0, **kwargs):
        assert (
            self.dim == 3
        ), "Shape must be 3-dimensional to realize inertial parameters."
        if not params.consistent(eps=eps):
            return False

        # non-degenerate case
        if not self.is_degenerate():
            assert np.isfinite(self.Q).all()
            return np.trace(self.Q @ params.J) >= 0  # TODO tolerance?

        # degenerate case requires checking Q matrices in the zero and non-zero
        # length directions separately
        # NOTE we do not check realizability for lower-dimensional ellipsoids;
        # it is supported in must_realize to facilitate building other shapes
        nonzero_Q, zero_Q = self._Q_degenerate()
        return np.trace(nonzero_Q @ params.J) >= 0 and np.isclose(
            np.trace(zero_Q @ params.J), 0
        )

    def must_realize(self, param_var, eps=0):
        assert (
            self.dim == 3
        ), "Shape must be 3-dimensional to realize inertial parameters."
        J, psd_constraints = pim_must_equal_param_var(param_var, eps)

        # don't need to do anything special for non-degenerate case
        if not self.is_degenerate():
            return psd_constraints + [cp.trace(self.Q @ J) >= 0]

        if USE_ELLIPSOID_TRACE_REALIZABILITY_CONSTRAINTS:
            nonzero_Q, zero_Q = self._Q_degenerate()
            return psd_constraints + [
                cp.trace(nonzero_Q @ J) >= 0,
                cp.trace(zero_Q @ J) == 0,
            ]
        else:
            r = self.rank
            Jr = cp.Variable((r + 1, r + 1), PSD=True)
            Qr, V = self._QV_degenerate()
            return psd_constraints + [cp.trace(Qr @ Jr) >= 0, J == V @ Jr @ V.T]

    def mbe(self, rcond=None, sphere=False):
        if not sphere:
            return self
        radius = np.max(self.half_extents)
        return Ellipsoid.sphere(radius=radius, center=self.center)

    def mbb(self):
        """Minimum-volume bounding box."""
        return Box(
            half_extents=self.half_extents,
            center=self.center,
            rotation=self.rotation,
        )

    def mib(self):
        """Maximum-volume inscribed box."""
        return Box(
            half_extents=self.half_extents / np.sqrt(3),
            center=self.center,
            rotation=self.rotation,
        )

    def aabb(self):
        v_max = self.center + np.sqrt(np.diag(self.E))
        v_min = self.center - np.sqrt(np.diag(self.E))
        return Box.from_two_vertices(v_min, v_max)

    def random_points(self, shape=1, rng=None):
        """Uniformly sample points inside the ellipsoid.

        See https://arxiv.org/abs/1404.1347
        """
        # uniformly sample points in the unit ball then affinely transform them
        # into the ellipsoid
        X = random_points_in_ball(shape=shape, dim=self.dim, rng=rng)
        Ainv = self.rotation @ np.diag(self.half_extents) @ self.rotation.T
        return X @ Ainv + self.center

    def random_points_on_surface(self, shape=1, rng=None):
        """Uniformly sample points on the surface of the ellipsoid.

        Only non-degenerate ellipsoids are supported.

        See Sec. 5.1 of https://doi.org/10.1007/s11075-023-01628-4
        """
        if np.isscalar(shape):
            shape = (shape,)
        n = np.prod(shape)  # total number of points to produce

        rng = np.random.default_rng(rng)

        assert np.all(
            self.half_extents > 0
        ), "Ellipsoid must be non-degenerate."
        d = 1.0 / self.half_extents**4

        m = np.min(self.half_extents)
        points = np.zeros((n, self.dim))
        count = 0
        while count < n:

            # sample as many points as we still need
            rem = n - count
            p = np.atleast_2d(
                random_points_on_hypersphere(
                    shape=rem, dim=self.dim - 1, rng=rng
                )
            )

            # scale to match the ellipsoid's axes
            r = p * self.half_extents

            # reject some points so that the overall sampling is uniform
            g = m * np.sqrt(np.sum(r**2 * d, axis=1))
            u = rng.random(rem)
            accept = u <= g
            n_acc = np.sum(accept)
            points[count : count + n_acc] = r[accept, :]
            count += n_acc

        # rotate and translate points as needed
        points = points @ self.rotation + self.center

        # reshape to desired shape
        if shape == (1,):
            return np.squeeze(points)
        return points.reshape(shape + (self.dim,))

    def transform(self, rotation=None, translation=None):
        rotation, translation = clean_transform(
            rotation=rotation, translation=translation, dim=self.dim
        )
        new_rotation = rotation @ self.rotation
        new_center = rotation @ self.center + translation
        half_extents = self.half_extents.copy()
        return Ellipsoid(
            half_extents=half_extents, rotation=new_rotation, center=new_center
        )

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
        new_rotation = rotation @ self.rotation
        half_extents = self.half_extents.copy()
        center = self.center.copy()
        return Ellipsoid(
            half_extents=half_extents, rotation=new_rotation, center=center
        )

    def contains_ellipsoid(self, other, solver=None):
        # See Boyd and Vandenberghe pp. 411
        # TODO does not work for degenerate ellipsoids
        t = cp.Variable(1)
        objective = cp.Minimize([0])  # feasibility problem
        constraints = [
            t >= 0,
            schur(
                self.S - t * other.A,
                self.A @ self.b - t * self.b,
                self.b @ self.b - 1 - t * (other.b @ other.b - 1),
            )
            << 0,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver)
        return problem.status == "optimal"

    def uniform_density_params(self, mass):
        """Generate the inertial parameters corresponding to a uniform mass density.

        The inertial parameters are generated with respect to the origin.

        Parameters
        ----------
        mass : float, non-negative
            The mass of the body.

        Returns
        -------
        : InertialParameters
            The inertial parameters.
        """
        assert mass >= 0, "Mass must be non-negative."

        H = mass * np.diag(self.half_extents**2) / 5.0
        return InertialParameters(mass=mass, h=np.zeros(3), H=H).transform(
            rotation=self.rotation, translation=self.center
        )

    def hollow_density_params(self, mass):
        """Generate the inertial parameters corresponding to an ellipsoid shell.

        In other words, all of the mass is uniformly distributed on the surface
        of the ellipsoid. The inertial parameters are generated with respect to
        the origin.

        Parameters
        ----------
        mass : float, non-negative
            The mass of the body.

        Returns
        -------
        : InertialParameters
            The inertial parameters.
        """
        assert mass >= 0, "Mass must be non-negative."

        x, y, z = self.half_extents
        d = 5 * (x * y + x * z + y * z)
        Hxx = x**2 * (x * y + x * z + 3 * y * z) / d
        Hyy = y**2 * (x * y + 3 * x * z + y * z) / d
        Hzz = z**2 * (3 * x * y + x * z + y * z) / d
        H = np.diag([Hxx, Hyy, Hzz])
        return InertialParameters(mass=mass, h=np.zeros(3), H=H).transform(
            rotation=self.rotation, translation=self.center
        )


def _mbee_con_mat(S, d, Ai, bi, ti):
    """Constraint matrix for minimum bounding ellipsoid of ellipsoids problem."""
    dim = S.shape[0]
    ci = bi @ bi - 1
    Z = np.zeros((dim, dim))
    f = cp.reshape(-1 - ti * ci, (1, 1))
    e = cp.reshape(d - ti * bi, (dim, 1))
    d = cp.reshape(d, (dim, 1))
    # fmt: off
    return cp.bmat([
        [S - ti * Ai, e, Z],
        [e.T, f, d.T],
        [Z, d, -S]])
    # fmt: on


def mbe_of_ellipsoids(ellipsoids, sphere=False, solver=None):
    """Compute the minimum-volume bounding ellipsoid for a set of ellipsoids.

    See :cite:t:`boyd2004convex`, Section 8.4.1.

    Parameters
    ----------
    ellipsoids : Iterable[Ellipsoid]
        The union of ellipsoids to bound.
    sphere : bool
        If ``True``, compute the minimum bounding *sphere*. Defaults to ``False``.
    solver : str or None
        The solver for cvxpy to use.

    Returns
    -------
    : Ellipsoid
        The minimum-volume bounding ellipsoid.
    """
    n = len(ellipsoids)
    dim = ellipsoids[0].dim

    S = cp.Variable((dim, dim), PSD=True)  # = A^2
    d = cp.Variable(dim)  # = Ab
    ts = cp.Variable(n)

    objective = cp.Minimize(-cp.log_det(S))
    constraints = [ts >= 0] + [
        _mbee_con_mat(S, d, E.A, E.b, ti) << 0 for E, ti in zip(ellipsoids, ts)
    ]
    if sphere:
        # if we want a sphere, then S is a multiple of the identity matrix
        a = cp.Variable(1)
        constraints.append(S == a * np.eye(dim))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    center = np.linalg.solve(S.value, -d.value)
    return Ellipsoid.from_shape_matrix(S.value, center=center)


def mbe_of_points(points, rcond=None, sphere=False, solver=None):
    """Compute the minimum-volume bounding ellipsoid for a set of points.

    See :cite:t:`boyd2004convex`, Section 8.4.1.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
        The points to bound. There are ``n`` points in ``d`` dimensions.
    rcond : float, optional
        Conditioning number used for internal routines.
    sphere : bool
        If ``True``, compute the minimum bounding *sphere*. Defaults to ``False``.
    solver : str or None,
        The solver for cvxpy to use.

    Returns
    -------
    : Ellipsoid
        The minimum-volume bounding ellipsoid.
    """
    # rowspace
    r = points[0]
    R = orth((points - r).T, rcond=rcond)
    rank = R.shape[1]

    # project onto the rowspace
    # this allows us to handle degenerate sets of points that live in a
    # lower-dimensional subspace than R^d
    P = (points - r) @ R

    # ellipsoid is parameterized as ||Ax + b|| <= 1 for the opt problem
    A = cp.Variable((rank, rank), PSD=True)
    b = cp.Variable(rank)

    objective = cp.Minimize(-cp.log_det(A))
    constraints = [cp.norm2(A @ x + b) <= 1 for x in P]
    if sphere:
        # if we want a sphere, then A is a multiple of the identity matrix
        r = cp.Variable(1)
        constraints.append(A == r * np.eye(rank))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    # unproject
    eigs, V = np.linalg.eigh(A.value)
    half_extents = np.zeros(points.shape[1])
    nz = np.nonzero(eigs)
    half_extents[nz] = 1.0 / eigs[nz]

    N = null_space((R @ V).T, rcond=rcond)
    rotation = np.hstack((R @ V, N))
    # if np.linalg.det(rotation) < 0:
    #     rotation = -rotation  # TODO
    center = R @ np.linalg.lstsq(A.value, -b.value, rcond=rcond)[0] + r

    return Ellipsoid(
        half_extents=half_extents, rotation=rotation, center=center
    )
