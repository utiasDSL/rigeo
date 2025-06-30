import numpy as np
import cvxpy as cp

from ..constraint import pim_must_equal_param_var
from ..polydd import SpanForm, FaceForm
from ..util import clean_transform, transform_matrix_inv, box_vertices
from ..inertial import InertialParameters
from .poly import ConvexPolyhedron


def _box_vertices(half_extents, rotation=None, center=None):
    """Generate the vertices of an oriented box."""
    rotation, center = clean_transform(rotation=rotation, translation=center)
    v = box_vertices(half_extents)
    return (rotation @ v.T).T + center


def _box_inequalities(half_extents, rotation=None, center=None):
    """Inequality form of an oriented box."""
    rotation, center = clean_transform(rotation=rotation, translation=center)

    x, y, z = half_extents
    A = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )
    b = np.array([x, x, y, y, z, z])

    A = A @ rotation.T
    b = b + A @ center
    return A, b


class Box(ConvexPolyhedron):
    """An oriented box/cuboid.

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

    # TODO change parameter order to match transform
    def __init__(self, half_extents, center=None, rotation=None):
        self.half_extents = np.array(half_extents)
        assert self.half_extents.shape == (3,)
        assert np.all(self.half_extents >= 0)

        self.rotation, self.center = clean_transform(
            rotation=rotation, translation=center
        )
        assert self.rotation.shape == (3, 3)
        assert self.center.shape == (3,)

        vertices = _box_vertices(
            self.half_extents, rotation=self.rotation, center=self.center
        )
        A, b = _box_inequalities(
            self.half_extents, rotation=self.rotation, center=self.center
        )
        span_form = SpanForm(vertices=vertices)
        face_form = FaceForm(A_ineq=A, b_ineq=b)

        super().__init__(span_form=span_form, face_form=face_form)

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
        """Check if points are on the surface of the box.

        Parameters
        ----------
        points : iterable
            Points to check. May be a single point or a list or array of points.
        tol : float, non-negative
            Numerical tolerance for qualifying as on the surface of the box.

        Returns
        -------
        :
            Given a single point, return ``True`` if the point is on the
            surface of the box, or ``False`` if not. For multiple points,
            return a boolean array with one value per point.
        """
        points = np.asarray(points)
        ndim = points.ndim
        assert (
            ndim <= 2
        ), f"points array must have at most 2 dims, but has {ndim}."
        points = np.atleast_2d(points)

        contained = self.contains(points, tol=tol)

        # for each point, check if at least one coordinate is on a face
        x, y, z = self.half_extents
        points = (points - self.center) @ self.rotation
        x_mask = np.isclose(np.abs(points[:, 0]), x, rtol=0, atol=tol)
        y_mask = np.isclose(np.abs(points[:, 1]), y, rtol=0, atol=tol)
        z_mask = np.isclose(np.abs(points[:, 2]), z, rtol=0, atol=tol)

        res = contained & (x_mask | y_mask | z_mask)
        if ndim == 1:
            return res[0]
        return res

    def grid(self, n):
        """Generate a set of points evenly spaced along each dimension in the box.

        Parameters
        ----------
        n : int or Iterable
            The number of points in each of the three dimensions. If a single
            int, then the number of points in each dimension is the same.
            Otherwise, a different value can be specified for each dimension.

        Returns
        -------
        :
            An array of points with shape ``(n**3, 3)``.
        """
        if np.isscalar(n):
            n = [n, n, n]

        L = -self.half_extents
        U = self.half_extents

        x = np.linspace(L[0], U[0], n[0])
        y = np.linspace(L[1], U[1], n[1])
        z = np.linspace(L[2], U[2], n[2])

        X, Y, Z = np.meshgrid(x, y, z)
        points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        return (self.rotation @ points.T).T + self.center

    def transform(self, rotation=None, translation=None):
        rotation, translation = clean_transform(
            rotation=rotation, translation=translation
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
        from .ellipsoid import Ellipsoid

        if sphere:
            radius = np.min(self.half_extents)
            return Ellipsoid.sphere(radius=radius, center=self.center)
        return Ellipsoid(
            half_extents=self.half_extents,
            center=self.center,
            rotation=self.rotation,
        )

    def mbe(self, rcond=None, sphere=False, solver=None):
        from .ellipsoid import Ellipsoid

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

    def moment_custom_vertex_constraints(self, param_var, eps=0):
        """My own custom constraints for boxes."""
        J, psd_constraints = pim_must_equal_param_var(param_var, eps)
        T = transform_matrix_inv(
            rotation=self.rotation, translation=self.center
        )

        # transform back to the origin
        J0 = T @ J @ T.T
        m = J0[3, 3]
        H = J0[:3, :3]

        # vertices of the axis-aligned box centered at the origin
        vs = _box_vertices(self.half_extents)
        Vs = [np.append(v, 1) for v in vs]

        μs = cp.Variable(8, nonneg=True)
        Jv = cp.sum([μ * np.outer(V, V) for μ, V in zip(μs, Vs)])
        mv = Jv[3, 3]
        Hv = Jv[:3, :3]

        return psd_constraints + [
            m == mv,
            cp.upper_tri(J0) == cp.upper_tri(Jv),
            cp.diag(H) <= cp.diag(Hv),
        ]

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
        H = mass * np.diag([Hxx, Hyy, Hzz])
        return InertialParameters(mass=mass, h=np.zeros(3), H=H).transform(
            rotation=self.rotation, translation=self.center
        )

    def wireframe_density_params(self, mass):
        """Generate the inertial parameters corresponding to a wireframe box.

        In other words, all of the mass is uniformly distributed on the edges
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
        d = x + y + z
        Hxx = x**2 * (x / 3 + y + z) / d
        Hyy = y**2 * (x + y / 3 + z) / d
        Hzz = z**2 * (x + y + z / 3) / d
        H = mass * np.diag([Hxx, Hyy, Hzz])
        return InertialParameters(mass=mass, h=np.zeros(3), H=H).transform(
            rotation=self.rotation, translation=self.center
        )
