import numpy as np
import cvxpy as cp
from scipy.linalg import orth, null_space

from ..util import clean_transform
from ..random import random_points_in_ball, random_points_on_hypersphere
from ..inertial import InertialParameters
from ..constraint import schur, pim_must_equal_param_var

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
        ndim = points.ndim
        assert ndim <= 2, f"points must have 1 or 2 dimensions, but has {ndim}."
        points = np.atleast_2d(points)

        z = np.isclose(self.half_extents, 0)
        nz = ~z
        s = self.half_extents_inv[nz] ** 2

        # transform back to the origin
        ps = (points - self.center) @ self.rotation
        pz = ps[:, z]
        pnz = ps[:, nz]

        # degenerate dimensions
        res_z = np.all(np.isclose(pz, 0, rtol=0, atol=tol), axis=1)

        # nondegenerate dimensions
        res_nz = np.sum((s * pnz) * pnz, axis=1) <= 1 + tol

        # combine them
        res = res_z & res_nz
        if ndim == 1:
            return res[0]
        return res

    def on_surface(self, points, tol=1e-8):
        # TODO
        points = np.atleast_2d(points)
        contained = self.contains(points, tol=tol)

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
        from .box import Box

        return Box(
            half_extents=self.half_extents,
            center=self.center,
            rotation=self.rotation,
        )

    def mib(self):
        """Maximum-volume inscribed box."""
        from .box import Box

        return Box(
            half_extents=self.half_extents / np.sqrt(3),
            center=self.center,
            rotation=self.rotation,
        )

    def aabb(self):
        from .box import Box

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


def mie_inequality_form(A, b, sphere=False, solver=None):
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


