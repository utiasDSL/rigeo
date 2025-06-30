import numpy as np
import cvxpy as cp
from scipy.linalg import orth, null_space

from ..polydd import SpanForm, FaceForm
from ..util import clean_transform
from ..constraint import schur, pim_must_equal_param_var
from ..random import random_weight_vectors
from ..inertial import InertialParameters
from ..moment import Polynomial, MomentMatrix
from .base import Shape


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

        self._polys = [Polynomial.affine(a=a, b=b) for a, b in zip(self.A, self.b)]

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

    @classmethod
    def simplex(cls, extents):
        """A :math:`d`-dimensional simplex.

        The simplex has :math:`d+1` vertices: :math:`\\boldsymbol{v}_0 =
        \\boldsymbol{0}`, :math:`\\boldsymbol{v}_1 = (e_1, 0, 0,\\dots)`,
        :math:`\\boldsymbol{v}_2 = (0, e_2, 0,\\dots)`, etc., where :math:`e_i`
        corresponds to ``extents[i]``.

        Parameters
        ----------
        extents : np.ndarray, shape (d,)
            The extents of the simplex.

        Returns
        -------
        : ConvexPolyhedron
            The simplex.
        """
        extents = np.array(extents)
        assert np.all(extents > 0), "Simplex extents must be positive."

        dim = extents.shape[0]
        vertices = np.vstack((np.zeros(dim), np.diag(extents)))
        return cls.from_vertices(vertices)

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
        return np.all(points @ self.A.T <= self.b + tol, axis=1)

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
        """Sample random points from inside the polyhedron.

        Note that this is *not* uniform sampling.
        """
        if np.isscalar(shape):
            shape = (shape,)
        full_shape = tuple(shape) + (self.nv,)
        w = random_weight_vectors(full_shape, rng=rng)
        points = w @ self.vertices
        if shape == (1,):
            return np.squeeze(points)
        return points

    def aabb(self):
        from .box import Box

        return Box.from_points_to_bound(self.vertices)

    def mbe(self, rcond=None, sphere=False, solver=None):
        """Construct the minimum-volume bounding ellipsoid for this polyhedron."""
        from .ellipsoid import mbe_of_points

        return mbe_of_points(
            self.vertices, rcond=rcond, sphere=sphere, solver=solver
        )

    def mie(self, rcond=None, sphere=False, solver=None):
        """Compute the maximum inscribed ellipsoid for a polyhedron represented by
        a set of vertices.

        Returns the ellipsoid.
        """
        from .ellipsoid import Ellipsoid, mie_inequality_form

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
        ell = mie_inequality_form(
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

    def moment_sdp_feasibility_problem(self, param_var, eps=0, d=2):
        objective = cp.Minimize([0])
        constraints = self.moment_sdp_constraints(param_var, eps=eps, d=d)
        return cp.Problem(objective, constraints)

    def moment_sdp_constraints(self, param_var, eps=0, d=2):
        """Moment-based constraints for density realizability."""
        assert (
            self.dim == 3
        ), "Shape must be 3-dimensional to realize inertial parameters."
        J, psd_constraints = pim_must_equal_param_var(param_var, eps)
        m = J[3, 3]
        h = J[:3, 3]
        H = J[:3, :3]

        moment = MomentMatrix(n=self.dim, d=d)
        M = cp.Variable(moment.shape, PSD=True)

        return (
            moment.moment_constraints(M)
            + moment.localizing_constraints(M, polys=self._polys)
            + [m == M[0, 0], h == M[0, 1:4], H == M[1:4, 1:4]]
        )

    def moment_lp_constraints(self, param_var, eps=0, d=2):
        raise NotImplementedError()

    def moment_vertex_constraints(self, param_var, eps=0):
        """My custom constraints for density realizability based on combinations of vertices."""
        assert (
            self.dim == 3
        ), "Shape must be 3-dimensional to realize inertial parameters."

        # TODO this could fairly easily be revised to not be limited to 3D
        J, psd_constraints = pim_must_equal_param_var(param_var, eps)
        m = J[3, 3]
        h = J[:3, 3]
        H = J[:3, :3]

        Vs = np.array([np.outer(v, v) for v in self.vertices])
        ms = cp.Variable(self.nv, nonneg=True)

        return psd_constraints + [
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

    def grid(self, n):
        box = self.aabb()
        grid = box.grid(n)
        idx = self.contains(grid)
        return grid[idx, :]
