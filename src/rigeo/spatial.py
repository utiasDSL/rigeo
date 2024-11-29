import numpy as np

from .util import skew3


class SpatialVector:
    """Spatial vector class, which contains a linear and an angular component."""

    def __init__(self, linear=None, angular=None):
        self.vec = np.zeros(6)
        if angular is not None:
            assert len(angular) == 3
            self.vec[:3] = angular
        if linear is not None:
            assert len(linear) == 3
            self.vec[3:] = linear

    @classmethod
    def from_vec(cls, vec):
        assert vec.shape == (6,)
        return cls(linear=vec[3:], angular=vec[:3])

    @classmethod
    def zero(cls):
        return cls()

    def __repr__(self):
        return f"SpatialVector(linear={self.linear}, angular={self.angular})"

    def __add__(self, other):
        return SpatialVector(
            linear=self.linear + other.linear,
            angular=self.angular + other.angular,
        )

    def __sub__(self, other):
        return SpatialVector(
            linear=self.linear - other.linear,
            angular=self.angular - other.angular,
        )

    def __mul__(self, other):
        return SpatialVector(
            linear=self.linear * other, angular=self.angular * other
        )

    def __truediv__(self, other):
        return SpatialVector(
            linear=self.linear / other, angular=self.angular / other
        )

    @property
    def linear(self):
        return self.vec[3:]

    @property
    def angular(self):
        return self.vec[:3]

    @staticmethod
    def random_diag_normal(mean, cov_diag, rng=None):
        rng = np.random.default_rng(rng)
        sample = rng.multivariate_normal(
            mean=mean.vec, cov=np.diag(cov_diag.vec)
        )
        return SpatialVector.from_vec(sample)

    def adjoint(self):
        Sv = skew3(self.linear)
        Sω = skew3(self.angular)
        return np.block([[Sω, np.zeros((3, 3))], [Sv, Sω]])


SV = SpatialVector
