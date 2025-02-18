import cvxpy as cp
import numpy as np

from ..constraint import pim_must_equal_param_var
from ..inertial import InertialParameters
from ..random import random_points_in_ball
from ..util import clean_transform, transform_matrix_inv
from .base import Shape
from .box import Box
from .ellipsoid import Ellipsoid


class Cylinder(Shape):
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

        # build the lower-dimensional ellipsoids for checking containment and
        # realizability
        ell1 = Ellipsoid(
            half_extents=self.length / 2,
            center=self.rotation[:, 2] @ self.center,
        )
        ell2 = Ellipsoid(
            half_extents=[self.radius, self.radius],
            center=self.rotation[:, :2].T @ self.center,
        )
        self._ellipsoids = [ell1, ell2]

        U1 = np.zeros((4, 2))
        U1[:3, 0] = self.rotation[:, 2]
        U1[3, 1] = 1

        U2 = np.zeros((4, 3))
        U2[:3, :2] = self.rotation[:, :2]
        U2[3, 2] = 1
        self._Us = [U1, U2]

        # disks that make up the convex hull
        # TODO can I just use one of ellipsoids or disks?
        h = 0.5 * self.length
        disk_extents = [self.radius, self.radius, 0]
        disk_centers = np.array([[0, 0, h], [0, 0, -h]])

        # disk_centers = self.endpoints()
        disk1 = Ellipsoid(
            half_extents=disk_extents,
            center=disk_centers[0],
            # rotation=self.rotation,
        )
        disk2 = Ellipsoid(
            half_extents=disk_extents,
            center=disk_centers[1],
            # rotation=self.rotation,
        )
        self.disk0s = [disk1, disk2]
        self.disks = [
            disk.transform(rotation=self.rotation, translation=self.center)
            for disk in self.disk0s
        ]

    @property
    def longitudinal_axis(self):
        return self.rotation[:, 2]

    @property
    def transverse_axes(self):
        return self.rotation[:, :2]

    @property
    def volume(self):
        """The volume of the cylinder."""
        return np.pi * self.radius**2 * self.length

    def is_same(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            np.isclose(self.length, other.length)
            and np.isclose(self.radius, other.radius)
            and np.allclose(self.center, other.center)
            and np.allclose(self.rotation, other.rotation)
        )

    def contains(self, points, tol=1e-8):
        P1 = points @ self.longitudinal_axis[:, None]
        P2 = points @ self.transverse_axes
        return np.all(
            [E.contains(P) for E, P in zip(self._ellipsoids, (P1, P2))], axis=0
        )

    def must_contain(self, points, scale=1.0):
        P1 = points @ self.longitudinal_axis[:, None]
        P2 = points @ self.transverse_axes
        return [
            c
            for E, P in zip(self._ellipsoids, (P1, P2))
            for c in E.must_contain(P, scale=scale)
        ]

    def moment_sdp_constraints(self, param_var, eps=0, d=2):
        raise NotImplementedError()

    def moment_cylinder_vertex_constraints(self, param_var, eps=0):
        J, psd_constraints = pim_must_equal_param_var(param_var, eps)
        T = transform_matrix_inv(
            rotation=self.rotation, translation=self.center
        )

        # transform back to the origin
        J0 = T @ J @ T.T
        m = J0[3, 3]
        H = J0[:3, :3]

        Js = [cp.Variable((4, 4), PSD=True) for _ in self.disk0s]
        J_sum = cp.Variable((4, 4), PSD=True)
        m_sum = J_sum[3, 3]

        # TODO can I actually realize the xx and yy H components exactly?
        return (
            psd_constraints
            + [
                J_sum == cp.sum(Js),
                m == m_sum,
                cp.upper_tri(J0) == cp.upper_tri(J_sum),
                H[0, 0] == J_sum[0, 0],
                H[1, 1] == J_sum[1, 1],
                H[2, 2] <= J_sum[2, 2],
            ]
            + [
                c
                for J, disk in zip(Js, self.disk0s)
                for c in disk.moment_constraints(J, eps=eps)
            ]
        )

        # + [
        #     J_sum == cp.sum(Js),
        #     J[3, 3] == J_sum[3, 3],
        #     J[:3, 3] == J_sum[:3, 3],
        #     J[:3, :3] << J_sum[:3, :3],
        # ]
        # + [
        #     c
        #     for J, disk in zip(Js, self.disks)
        #     for c in disk.moment_constraints(J, eps=0)
        # ]

    def transform(self, rotation=None, translation=None):
        rotation, translation = clean_transform(
            rotation=rotation, translation=translation, dim=3
        )
        new_rotation = rotation @ self.rotation
        new_center = rotation @ self.center + translation
        return Cylinder(
            length=self.length,
            radius=self.radius,
            rotation=new_rotation,
            center=new_center,
        )

    def endpoints(self):
        """Get the two points at the ends of the longitudinal axis."""
        h = 0.5 * self.length * self.longitudinal_axis
        return np.array([h, -h]) + self.center

    def aabb(self):
        c = self.endpoints()
        disk1 = Ellipsoid(
            half_extents=[self.radius, self.radius, 0],
            center=c[0],
            rotation=self.rotation,
        )
        disk2 = Ellipsoid(
            half_extents=[self.radius, self.radius, 0],
            center=c[1],
            rotation=self.rotation,
        )
        points = np.vstack((disk1.aabb().vertices, disk2.aabb().vertices))
        return Box.from_points_to_bound(points)

    def mbe(self, rcond=None, sphere=False, solver=None):
        return self.mib().mbe(rcond=rcond, sphere=sphere, solver=solver)

    def random_points(self, shape=1, rng=None):
        rng = np.random.default_rng(rng)
        P_z = self.length * rng.uniform(low=-0.5, high=0.5, size=shape)
        P_xy = self.radius * random_points_in_ball(shape=shape, dim=2, rng=rng)
        if shape != 1:
            P_z = np.expand_dims(P_z, axis=-1)
        P = np.concatenate((P_xy, P_z), axis=-1)
        return P @ self.rotation.T + self.center

    def mib(self):
        r = self.radius / np.sqrt(2)
        half_extents = [r, r, self.length / 2]
        return Box(
            half_extents=half_extents,
            rotation=self.rotation,
            center=self.center,
        )

    def mbb(self):
        half_extents = [self.radius, self.radius, self.length / 2]
        return Box(
            half_extents=half_extents,
            rotation=self.rotation,
            center=self.center,
        )

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

        Hxy = mass * self.radius**2 / 4.0
        Hz = mass * self.length**2 / 12.0
        H = np.diag([Hxy, Hxy, Hz])
        return InertialParameters(mass=mass, h=np.zeros(3), H=H).transform(
            rotation=self.rotation, translation=self.center
        )

    def capsule(self):
        """Constuct a capsule from this cylinder."""
        # local import needed to avoid circular import
        from .capsule import Capsule

        return Capsule(self)
