import numpy as np

from ..polydd import SpanForm
from ..util import clean_transform
from ..inertial import InertialParameters
from .poly import ConvexPolyhedron


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
        points = np.array(points)
        ndim = points.ndim
        assert ndim <= 2
        points = np.atleast_2d(points)
        contained = self.contains(points, tol=tol)

        # for each point, check if at least one coordinate is on a face
        # TODO use tol
        x, y, z = self.half_extents
        points = (points - self.center) @ self.rotation
        x_mask = np.isclose(np.abs(points[:, 0]), x)
        y_mask = np.isclose(np.abs(points[:, 1]), y)
        z_mask = np.isclose(np.abs(points[:, 2]), z)

        res = contained & (x_mask | y_mask | z_mask)
        if ndim == 1:
            return res[0]
        return res

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
        H = np.diag([Hxx, Hyy, Hzz])
        return InertialParameters(mass=mass, h=np.zeros(3), H=H).transform(
            rotation=self.rotation, translation=self.center
        )