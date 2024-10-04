import numpy as np

from .util import skew3


class SV:
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

    def __repr__(self):
        return f"SV(linear={self.linear}, angular={self.angular})"

    @property
    def linear(self):
        return self.vec[3:]

    @property
    def angular(self):
        return self.vec[:3]

    def adjoint(self):
        Sv = skew3(self.linear)
        Sω = skew3(self.angular)
        return np.block([[Sω, np.zeros((3, 3))], [Sv, Sω]])
