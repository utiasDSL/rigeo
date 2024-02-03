"""Polyhedral and ellipsoidal geometry."""
import abc
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import cvxpy as cp
from scipy.linalg import orth

from inertial_params.polydd import SpanForm, FaceForm
from inertial_params.util import schur
from inertial_params.random import random_weight_vectors
from inertial_params.inertial import pim_sum_vec_matrices, InertialParameters


def _inv_with_zeros(a, tol=1e-8):
    # a = a.copy()
    # mask = np.isclose(a, 0)
    # a[mask] = 1.0  # placeholder value
    # b = 1.0 / a
    # b[mask] = np.inf  # replace placeholders with inf
    # return b

    zero_mask = np.isclose(a, 0, rtol=0, atol=tol)
    out = np.inf * np.ones_like(a)
    np.divide(1.0, a, out=out, where=~zero_mask)
    return out


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


# TODO
class Shape(abc.ABC):
    @abc.abstractmethod
    def contains(self, points, tol):
        pass

    @abc.abstractmethod
    def must_contain(self):
        pass

    @abc.abstractmethod
    def can_realize(self):
        pass

    @abc.abstractmethod
    def must_realize(self):
        pass

    @abc.abstractmethod
    def minimum_bounding_box(self):
        pass

    @abc.abstractmethod
    def minimum_bounding_ellipsoid(self):
        pass

    @abc.abstractmethod
    def transform(self):
        pass


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
        """Test if the polyhedron contains a set of points.

        Parameters
        ----------
        points : np.ndarray, shape (n, self.dim)
            The points to check.
        tol : float, non-negative
            The numerical tolerance for membership.

        Returns
        -------
        : bool or np.ndarray of bool, shape (n,)
            Boolean array where each entry is ``True`` if the polyhedron
            contains the corresponding point and ``False`` otherwise.
        """
        points = np.array(points)
        if points.ndim == 1:
            return np.all(self.A @ points <= self.b + tol)
        return np.array([np.all(self.A @ p <= self.b + tol) for p in points])

    def must_contain(self, points, scale=1.0):
        """Generate cvxpy constraints to keep the points inside the polyhedron.

        Parameters
        ----------
        points : cp.Variable, shape (self.dim,) or (n, self.dim)
            A point or set of points to constrain to lie inside the polyhedron.
        scale : float, positive
            Scale for ``points``. The main idea is that one may wish to check
            that the CoM belongs to the shape, but using the quantity
            :math:`h=mc`. Then ``must_contain(c)`` is equivalent to
            ``must_contain(h, scale=m)``.

        Returns
        -------
        : list
            A list of cxvpy constraints that keep the points inside the polyhedron.
        """
        if points.ndim == 1:
            points = [points]
        return [self.A @ p <= scale * self.b for p in points]

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
        : Box
            The axis-aligned bounding box.
        """
        return Box.from_points_to_bound(self.vertices)

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
        # TODO it would be better to find the minimum oriented bounding box
        box_grid = self.aabb().grid(n)
        contained = self.contains(box_grid)
        return box_grid[contained, :]

    def minimum_bounding_ellipsoid(self, rcond=None, sphere=False):
        """Construct the minimum-volume bounding ellipsoid for this polyhedron."""
        return minimum_bounding_ellipsoid(self.vertices, rcond=rcond, sphere=sphere)

    def maximum_inscribed_ellipsoid(self, rcond=None, sphere=False):
        """Construct the maximum-volume ellipsoid inscribed in this polyhedron."""
        return maximum_inscribed_ellipsoid(self.vertices, rcond=rcond, sphere=sphere)

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

        return InertialParameters.from_point_masses(masses=masses, points=self.vertices)


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

        self._ellipsoids = []
        for i, r in enumerate(self.half_extents):
            half_extents = np.inf * np.ones(self.dim)
            half_extents[i] = r
            ell = Ellipsoid(
                half_extents=half_extents, rotation=self.rotation, center=self.center
            )
            self._ellipsoids.append(ell)

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
        return self._ellipsoids

    def transform(self, translation=None, rotation=None):
        """Apply a rigid transformation to the box."""
        center = self.center.copy()
        orn = self.rotation.copy()

        if rotation is not None:
            center = rotation @ center
            orn = rotation @ orn
        if translation is not None:
            center += translation
        return Box(half_extents=self.half_extents.copy(), center=center, rotation=orn)

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
        rotation = rotation @ self.rotation
        center = self.center.copy()
        half_extents = self.half_extents.copy()
        return Box(half_extents=half_extents, center=center, rotation=rotation)

    def can_realize(self, params):
        """Check if the shape can realize the inertial parameters.

        Parameters
        ----------
        params : InertialParameters
            The inertial parameters to check.

        Returns
        -------
        : bool
            ``True`` if the parameters are realizable, ``False`` otherwise.
        """
        Es = self.as_ellipsoidal_intersection()
        return np.all([np.trace(E.Q @ params.J) >= 0 for E in Es])

    def must_realize(self, param_var, eps=0):
        """Generate cvxpy constraints for inertial parameters to be realizable.

        Parameters
        ----------
        param_var : cp.Expression, shape (4, 4) or shape (10,)
            The cvxpy inertial parameter variable. If shape is ``(4, 4)``, this
            is interpreted as the pseudo-inertia matrix. If shape is ``(10,)``,
            this is interpreted as the inertial parameter vector.
        eps : float, non-negative
            Pseudo-inertia matrix ``J`` is constrained such that ``J - eps *
            np.eye(4)`` is positive semidefinite.

        Returns
        -------
        : list
            List of cvxpy constraints.
        """
        assert eps >= 0
        if param_var.shape == (4, 4):
            J = param_var
        elif param_var.shape == (10,):
            As = pim_sum_vec_matrices()
            J = cp.sum([A * p for A, p in zip(As, param_var)])
        else:
            raise ValueError(
                f"Parameter variable has unexpected shape {param_var.shape}"
            )

        Es = self.as_ellipsoidal_intersection()
        return [cp.trace(E.Q @ J) >= 0 for E in Es] + [J == J.T, J >> eps * np.eye(4)]

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

        # TODO could replace this with InertialParameters.transform
        R = self.rotation
        Hc = mass * np.diag(self.half_extents**2) / 3.0
        H = R @ Hc @ R.T + mass * np.outer(self.center, self.center)
        h = mass * self.center
        return InertialParameters(mass=mass, h=h, H=H)


class Ellipsoid:
    """Ellipsoid in ``dim`` dimensions.

    The ellipsoid may be degenerate in two ways:
    1. If one or more half extents is infinite, then the ellipsoid is unbounded
       along one or more axes.
    2. If one or more half extents is zero, then the ellipsoid actually lives
       in a lower-dimensional subspace.
    """

    def __init__(self, half_extents, rotation=None, center=None):
        self.half_extents = np.array(half_extents)
        assert np.all(self.half_extents >= 0)

        self.half_extents_inv = _inv_with_zeros(self.half_extents)

        if rotation is None:
            self.rotation = np.eye(self.dim)
        else:
            self.rotation = np.array(rotation)
        assert self.rotation.shape == (self.dim, self.dim)

        if center is None:
            self.center = np.zeros(self.dim)
        else:
            self.center = np.array(center)
        assert self.center.shape == (self.dim,)

    @property
    def dim(self):
        return self.half_extents.shape[0]

    @property
    def Einv(self):
        return self.rotation @ np.diag(self.half_extents_inv**2) @ self.rotation.T

    @property
    def E(self):
        return self.rotation @ np.diag(self.half_extents**2) @ self.rotation.T

    @property
    def rank(self):
        return self.dim - np.sum(np.isinf(self.half_extents))

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
    def from_Einv(cls, Einv, center=None):
        # we can use eigh since Einv is symmetric
        # TODO is V always a valid rotation matrix?
        eigs, V = np.linalg.eigh(Einv)

        half_extents_inv = np.sqrt(eigs)
        half_extents = _inv_with_zeros(half_extents_inv)

        return cls(half_extents=half_extents, rotation=V, center=center)

    @classmethod
    def from_Ab(cls, A, b, rcond=None):
        Einv = A @ A

        # use least squares instead of direct solve in case we have a
        # degenerate ellipsoid
        center = np.linalg.lstsq(A, -b, rcond=rcond)[0]
        return cls.from_Einv(Einv=Einv, center=center)

    @classmethod
    def from_Q(cls, Q):
        assert Q.shape[0] == Q.shape[1]
        dim = Q.shape[0] - 1
        Einv = -Q[:dim, :dim]
        q = Q[:dim, dim]
        center = np.linalg.solve(Einv, q)
        return cls.from_Einv(Einv=Einv, center=center)

    @property
    def A(self):
        """``A`` from ``(A, b)`` representation of the ellipsoid.

        .. math::
           \\mathcal{E} = \\{x\\in\\mathbb{R}^d \\mid \\|Ax+b\\|^2\\leq 1\\}
        """
        # TODO ensure this is tested properly
        return self.rotation @ np.diag(self.half_extents_inv) @ self.rotation.T

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

    def is_same(self, other):
        """Check if this ellipsoid is the same as another."""
        return np.allclose(self.Q, other.Q)

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
        Einv_diag = np.diag(self.half_extents_inv[~zero_mask] ** 2)

        if points.ndim == 1:
            p = self.rotation.T @ (points - self.center)

            # value along degenerate dimension must be zero
            if not np.allclose(p[zero_mask], 0, rtol=0, atol=tol):
                return False

            return p[~zero_mask] @ Einv_diag @ p[~zero_mask] <= 1 + tol
        elif points.ndim == 2:
            ps = (points - self.center) @ self.rotation

            # degenerate dimensions
            res1 = np.all(np.isclose(ps[:, zero_mask], 0, rtol=0, atol=tol), axis=1)

            # nondegenerate dimensions
            res2 = np.array(
                [p[~zero_mask] @ Einv_diag @ p[~zero_mask] <= 1 + tol for p in ps]
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

    def can_realize(self, params):
        # TODO
        assert self.dim == 3

    def must_realize(self, params):
        # TODO
        assert self.dim == 3

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
        if rotation is None:
            rotation = np.eye(self.dim)
        if translation is None:
            translation = np.zeros(self.dim)

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

    def uniform_density_params(self, mass):
        pass

    def hollow_density_params(self, mass):
        pass


class Cylinder:
    """A cylinder in three dimensions.

    Parameters
    ----------
    length : float, non-negative
        The length along the longitudinal axis.
    radius : float, non-negative
        The radius of the transverse cross-section.
    rotation : np.ndarray, shape (3, 3)
        Rotation matrix, where identity means the z-axis is the longitudinal
        axis.
    center : np.ndarray, shape (3,)
        The center of the cylinder. If not provided, defaults to the origin.
    """

    def __init__(self, length, radius, rotation=None, center=None):
        assert length >= 0
        assert radius >= 0

        self.length = length
        self.radius = radius

        if rotation is None:
            self.rotation = np.eye(3)
        else:
            self.rotation = np.array(rotation)
        assert self.rotation.shape == (3, 3)

        if center is None:
            self.center = np.zeros(3)
        else:
            self.center = np.array(center)
        assert self.center.shape == (3,)

        ell1 = Ellipsoid(
            half_extents=[self.radius, self.radius, np.inf],
            center=self.center,
            rotation=self.rotation,
        )
        ell2 = Ellipsoid(
            half_extents=[np.inf, np.inf, self.length / 2],
            center=self.center,
            rotation=self.rotation,
        )
        self._ellipsoids = [ell1, ell2]

    @property
    def longitudinal_axis(self):
        return self.rotation[:, 2]

    @property
    def transverse_axes(self):
        return self.rotation[:, :2]

    def as_ellipsoidal_intersection(self):
        """Construct a set of ellipsoids, the intersection of which is the cylinder."""
        return self._ellipsoids

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
        return np.all([E.contains(points) for E in self._ellipsoids], axis=0)

    def must_contain(self, points, scale=1.0):
        return [
            c for E in self._ellipsoids for c in E.must_contain(points, scale=scale)
        ]

    def inscribed_box(self):
        # TODO need tests for these
        r = self.radius / np.sqrt(2)
        half_extents = [r, r, self.length / 2]
        return Box(
            half_extents=half_extents, rotation=self.rotation, center=self.center
        )

    def bounding_box(self):
        half_extents = [self.radius, self.radius, self.length / 2]
        return Box(
            half_extents=half_extents, rotation=self.rotation, center=self.center
        )


class Capsule:
    """A capsule in three dimensions.

    Parameters
    ----------
    length : float, non-negative
        The length along the longitudinal axis.
    radius : float, non-negative
        The radius of the transverse cross-section.
    rotation : np.ndarray, shape (3, 3)
        Rotation matrix, where identity means the z-axis is the longitudinal
        axis.
    center : np.ndarray, shape (3,)
        The center of the cylinder. If not provided, defaults to the origin.
    """

    def __init__(self, length, radius, rotation=None, center=None):
        # TODO same API as the cylinder
        pass


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

    # qhull does not handle degenerate sets of points but cdd does, which is
    # nice
    return SpanForm(points).canonical().vertices


def cube_bounding_ellipsoid(half_extent):
    """Minimum-volume bounding ellipsoid (sphere) of a cube.

    Parameters
    ----------
    half_extent : float, non-negative
        The half extent of the cube (i.e., half of the side length).

    Returns the bounding ellipsoid.
    """
    radius = np.linalg.norm(half_extent * np.ones(3))
    return Ellipsoid.sphere(radius)


def cube_inscribed_ellipsoid(half_extent):
    """Maximum-volume inscribed ellipsoid (sphere) of a cube with half length h.

    Returns the ellipsoid.
    """
    return Ellipsoid.sphere(half_extent)


def minimum_bounding_ellipsoid(points, rcond=None, sphere=False):
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
    if sphere:
        # if we want a sphere, then A is a multiple of the identity matrix
        r = cp.Variable(1)
        constraints.append(A == r * np.eye(dim))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    # unproject back into the original space
    A = R @ A.value @ R.T
    b = R @ b.value
    return Ellipsoid.from_Ab(A=A, b=b, rcond=rcond)


# TODO add solver argument
def _maximum_inscribed_ellipsoid_inequality_form(A, b, sphere=False):
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
    if sphere:
        # if we want a sphere, then A is a multiple of the identity matrix
        r = cp.Variable(1)
        constraints.append(B == r * np.eye(dim))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    E = B.value @ B.value
    return Ellipsoid.from_Einv(Einv=np.linalg.inv(E), center=c.value)


def maximum_inscribed_ellipsoid(vertices, rcond=None, sphere=False):
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
    ell = _maximum_inscribed_ellipsoid_inequality_form(
        face_form.A, face_form.b, sphere=sphere
    )

    # unproject back into the original space
    Einv = R @ ell.Einv @ R.T
    center = R @ ell.center
    return Ellipsoid.from_Einv(Einv=Einv, center=center)


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


@dataclass
class ClosestPointInfo:
    """Information about a closest point query.

    Attributes
    ----------
    p1 : np.ndarray
        The closest point on the first shape.
    p2 : np.ndarray
        The closest point on the second shape.
    dist : float, non-negative
        The distance between the two shapes.
    """

    p1: np.ndarray
    p2: np.ndarray
    dist: float


def closest_points(shape1, shape2, solver=None):
    """Compute the closest points between two shapes.

    When the two shapes are in contact or penetrating, the distance will be
    zero and the points can be anything inside the intersection.

    This function is *not* optimized for speed: a full convex program is
    solved. Useful for prototyping but not for high-speed queries.

    Parameters
    ----------
    shape1 : ConvexPolyhedron or Ellipsoid or Cylinder
        The first shape to check.
    shape2 : ConvexPolyhedron or Ellipsoid or Cylinder
        The second shape to check.
    solver : str or None
        The solver for cvxpy to use.

    Returns
    -------
    : ClosestPointInfo
        Information about the closest points.
    """
    p1 = cp.Variable(3)
    p2 = cp.Variable(3)

    objective = cp.Minimize(cp.norm2(p2 - p1))
    constraints = shape1.must_contain(p1) + shape2.must_contain(p2)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    return ClosestPointInfo(p1=p1.value, p2=p2.value, dist=objective.value)
