import numpy as np

import rigeo.util as util
from rigeo.random import random_psd_matrix


def H2I(H):
    """Convert second moment matrix to inertia matrix."""
    assert H.shape == (3, 3)
    return np.trace(H) * np.eye(3) - H


def I2H(I):
    """Convert inertia matrix to second moment matrix."""
    assert I.shape == (3, 3)
    return 0.5 * np.trace(I) * np.eye(3) - I


class InertialParameters:
    """Inertial parameters of a rigid body.

    Parameters
    ----------
    mass : float
        Mass of the body.
    h : np.ndarry, shape (3,)
        First moment of mass.
    H : np.ndarray, shape (3, 3)
        Second moment matrix.
    """

    def __init__(self, mass, h, H):
        assert mass >= 0, f"Mass must be non-negative but is {mass}."

        self.mass = mass
        self.h = h
        self.H = H

        # min_Hc_λ = np.min(np.linalg.eigvals(self.Hc))
        # assert min_Hc_λ >= -tol, f"Hc must be p.s.d. but min eigenval is {min_H_λ}"

    def __repr__(self):
        return f"InertialParameters(mass={self.mass}, h={self.h}, H={self.H})"

    @classmethod
    def zero(cls):
        return cls(mass=0, h=np.zeros(3), H=np.zeros((3, 3)))

    @property
    def com(self):
        """Center of mass."""
        return self.h / self.mass

    @property
    def θ(self):
        """Inertial parameter vector."""
        return np.concatenate([[self.mass], self.h, util.vech(self.I)])

    @property
    def I(self):
        """Inertia matrix."""
        return H2I(self.H)

    @property
    def Hc(self):
        """H matrix about the CoM."""
        return self.H - np.outer(self.h, self.h) / self.mass

    @property
    def Ic(self):
        """Inertia matrix about the CoM."""
        return H2I(self.Hc)

    @property
    def J(self):
        """Pseudo-inertia matrix."""
        J = np.zeros((4, 4))
        J[:3, :3] = self.H
        J[:3, 3] = self.h
        J[3, :3] = self.h
        J[3, 3] = self.mass
        return J

    @property
    def M(self):
        """Spatial mass matrix."""
        S = util.skew3(self.h)
        return np.block([[self.mass * np.eye(3), -S], [S, self.I]])

    @classmethod
    def from_vector(cls, θ):
        """Construct from a parameter vector."""
        assert θ.shape == (10,)
        mass = θ[0]
        h = θ[1:4]
        # fmt: off
        I = np.array([
            [θ[4], θ[5], θ[6]],
            [θ[5], θ[7], θ[8]],
            [θ[6], θ[8], θ[9]]
        ])
        # fmt: on
        return cls(mass=mass, h=h, H=I2H(I))

    @classmethod
    def from_pseudo_inertia_matrix(cls, J):
        """Construct from a pseudo-inertia matrix."""
        assert J.shape == (4, 4)
        H = J[:3, :3]
        h = J[3, :3]
        mass = J[3, 3]
        return cls(mass=mass, h=h, H=H)

    @classmethod
    def from_point_masses(cls, masses, points):
        """Construct from a system of point masses."""
        masses = np.array(masses)
        points = np.array(points)
        assert masses.shape[0] == points.shape[0]

        # homogeneous representation
        P = np.hstack((points, np.ones((points.shape[0], 1))))

        J = sum([m * np.outer(p, p) for m, p in zip(masses, P)])
        return cls.from_pseudo_inertia_matrix(J)

    @classmethod
    def translate_from_com(cls, mass, h, Hc):
        """Construct from mass, h, and H matrix about the CoM.

        The H matrix is adjusted to be about the reference point defined by h.
        """
        assert mass > 0
        assert h.shape == (3,)
        assert Hc.shape == (3, 3)
        H = Hc + np.outer(h, h) / mass
        return cls(mass=mass, h=h, H=H)

    @classmethod
    def from_mcI(cls, mass, com, I):
        """Construct from mass, center of mass, and inertia matrix."""
        assert mass > 0
        assert com.shape == (3,)
        assert I.shape == (3, 3)
        h = com / mass
        H = I2H(I)
        return cls(mass=mass, h=h, H=H)

    @classmethod
    def random(cls):
        """Generate a random set of physically consistent inertial parameters.

        Useful for testing purposes.
        """
        mass = 0.1 + np.random.random() * 0.9
        com = np.random.random(3) - 0.5
        h = mass * com
        H = random_psd_matrix((3, 3)) + mass * np.outer(com, com)
        return cls(mass=mass, h=h, H=H)

    def is_same(self, other):
        """Check if this set of inertial parameters is the same as another.

        Parameters
        ----------
        other : InertialParameters
            The other set of inertial parameters to check.

        Returns
        -------
        : bool
            ``True`` if they are the same, ``False`` otherwise.
        """
        return np.allclose(self.J, other.J)

    def consistent(self, tol=1e-8):
        """Check if the inertial parameters are fully physically consistent.

        This means that there exists a physical rigid body that can have these
        parameters.

        Returns
        -------
        : bool
            ``True`` if the parameters are consistent, ``False`` otherwise.
        """
        return np.min(np.linalg.eigvals(self.J)) >= -tol

    def transform(self, rotation=None, translation=None):
        if rotation is None:
            rotation = np.eye(3)
        if translation is None:
            translation = np.zeros(3)

        h = rotation @ self.h + self.mass * translation
        H = rotation @ self.Hc @ rotation.T + np.outer(h, h) / self.mass
        return InertialParameters(mass=self.mass, h=h, H=H)

    def __add__(self, other):
        return InertialParameters(
            mass=self.mass + other.mass, h=self.h + other.h, H=self.H + other.H
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def body_wrench(self, V, A):
        """Compute the body-frame wrench about the reference point."""
        M = self.M
        return M @ A + util.skew6(V) @ M @ V
