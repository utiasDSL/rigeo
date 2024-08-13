import cvxpy as cp
import numpy as np

from ..random import rejection_sample
from .base import Shape
from .box import Box
from .ellipsoid import Ellipsoid


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
        new_cylinder = self.cylinder.transform(
            rotation=rotation, translation=translation
        )
        return Capsule(cylinder=new_cylinder)

    def can_realize(self, params, eps=0, **kwargs):
        if not params.consistent(eps=eps):
            return False

        J = cp.Variable((4, 4), PSD=True)

        objective = cp.Minimize([0])  # feasibility problem
        constraints = self.must_realize(J) + [J == params.J]
        problem = cp.Problem(objective, constraints)
        problem.solve(**kwargs)
        return problem.status == "optimal"

    def must_realize(self, param_var, eps=0):
        J, psd_constraints = pim_must_equal_param_var(param_var, eps)

        Js = [cp.Variable((4, 4), PSD=True) for _ in self.caps]
        J_sum = cp.Variable((4, 4), PSD=True)

        return (
            psd_constraints
            + [
                J_sum == cp.sum(Js),
                J[3, 3] == J_sum[3, 3],
                J[:3, 3] == J_sum[:3, 3],
                J[:3, :3] << J_sum[:3, :3],
            ]
            + [
                c
                for J, cap in zip(Js, self.caps)
                for c in cap.must_realize(J, eps=0)
            ]
        )

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
        mbb = self.mbb()
        return rejection_sample(
            actual_shapes=[self],
            bounding_shape=self.mbb(),
            sample_shape=shape,
            rng=rng,
        )
