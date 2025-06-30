import cvxpy as cp
import numpy as np

from ..random import rejection_sample
from .base import Shape
from .box import Box
from .ellipsoid import Ellipsoid, mbe_of_ellipsoids


class Capsule(Shape):
    """A capsule in three dimensions.

    Parameters
    ----------
    cylinder : Cylinder
        The cylinder to build the capsule from.
    """

    def __init__(self, cylinder):
        self.cylinder = cylinder
        self.caps = [
            Ellipsoid.sphere(radius=cylinder.radius, center=end)
            for end in cylinder.endpoints()
        ]

        self._shapes = self.caps + [self.cylinder]

    @property
    def center(self):
        return self.cylinder.center

    @property
    def rotation(self):
        return self.cylinder.rotation

    @property
    def inner_length(self):
        return self.cylinder.length

    @property
    def radius(self):
        return self.cylinder.radius

    @property
    def full_length(self):
        return self.inner_length + 2 * self.radius

    def is_same(self, other, tol=1e-8):
        if not isinstance(other, self.__class__):
            return False
        return self.cylinder.is_same(other.cylinder, tol=tol)

    def contains(self, points, tol=1e-8):
        contains = [shape.contains(points, tol=tol) for shape in self._shapes]
        return np.any(contains, axis=0)

    def must_contain(self, points, scale=1.0):
        if points.ndim == 1:
            points = [points]

        constraints = []
        for point in points:
            p1 = cp.Variable(3)
            p2 = cp.Variable(3)
            t = cp.Variable(1)
            constraints.extend(
                self.caps[0].must_contain(p1, scale=t)
                + self.caps[1].must_contain(p2, scale=scale - t)
                + [t >= 0, t <= scale, point == p1 + p2]
            )
        return constraints

    def transform(self, rotation=None, translation=None):
        return self.cylinder.transform(
            rotation=rotation, translation=translation
        ).capsule()

    def aabb(self):
        points = np.vstack([cap.aabb().vertices for cap in self.caps])
        return Box.from_points_to_bound(points)

    def mbe(self, rcond=None, sphere=False, solver=None):
        return mbe_of_ellipsoids(self.caps, sphere=sphere, solver=solver)

    def mbb(self):
        half_extents = [self.radius, self.radius, self.full_length / 2]
        return Box(
            half_extents=half_extents,
            rotation=self.rotation,
            center=self.center,
        )

    def random_points(self, shape=1, rng=None):
        # use rejection sampling within the bounding box
        return rejection_sample(
            actual_shapes=[self],
            bounding_shape=self.mbb(),
            sample_shape=shape,
            rng=rng,
        )
