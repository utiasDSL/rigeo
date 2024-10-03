import numpy as np

from .util import skew3, skew6, vech
from .random import random_psd_matrix


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

    Exactly one of ``h`` and ``com`` must be specified. Exactly one of ``H``
    and ``I`` must be specified.

    Parameters
    ----------
    mass : float
        Mass of the body.
    h : np.ndarray, shape (3,)
        First moment of mass.
    com : np.ndarray, shape (3,)
        Center of mass.
    H : np.ndarray, shape (3, 3)
        Second moment matrix.
    I : np.ndarray, shape (3, 3)
        Inertia matrix.
    translate_from_com : bool
        If ``True``, ``I``/``H`` will assumed to be about the center of mass,
        and will be translated accordingly to the origin.
    """

    def __init__(self, mass, h=None, com=None, H=None, I=None, translate_from_com=False):
        assert mass >= 0, f"Mass must be non-negative but is {mass}."
        self.mass = mass

        if h is not None and com is not None:
            raise ValueError("Cannot specify both h and com.")
        if com is None:
            if h is None:
                com = np.zeros(3)
            else:
                com = h / mass
        self.com = com

        if H is not None and I is not None:
            raise ValueError("Cannot specify both H and I.")
        if H is None:
            if I is None:
                if translate_from_com:
                    H = np.zeros((3, 3))
                else:
                    H = mass * np.outer(com, com)
            else:
                H = I2H(I)

        # translate the second moment matrix from CoM reference point to the
        # origin
        if translate_from_com:
            H = H + mass * np.outer(com, com)

        self.H = H

    def __repr__(self):
        return f"InertialParameters(mass={self.mass}, com={self.com}, H={self.H})"

    @classmethod
    def zero(cls):
        return cls(mass=0, com=np.zeros(3), H=np.zeros((3, 3)))

    @property
    def h(self):
        return self.mass * self.com

    @property
    def vec(self):
        """Inertial parameter vector."""
        return np.concatenate([[self.mass], self.h, vech(self.I)])

    @property
    def I(self):
        """Inertia matrix."""
        return H2I(self.H)

    @property
    def Hc(self):
        """H matrix about the CoM."""
        return self.H - self.mass * np.outer(self.com, self.com)

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
        S = skew3(self.h)
        return np.block([[self.mass * np.eye(3), -S], [S, self.I]])

    @classmethod
    def from_vec(cls, vec):
        """Construct from a parameter vector."""
        assert vec.shape == (10,)
        mass = vec[0]
        h = vec[1:4]
        # fmt: off
        I = np.array([
            [vec[4], vec[5], vec[6]],
            [vec[5], vec[7], vec[8]],
            [vec[6], vec[8], vec[9]]
        ])
        # fmt: on
        return cls(mass=mass, h=h, I=I)

    @classmethod
    def from_pim(cls, J):
        """Construct from a pseudo-inertia matrix.

        The pseudo-inertia matrix is

        .. math::
            \\boldsymbol{J} = \\begin{bmatrix}
                \\boldsymbol{H} & m\\boldsymbol{c} \\\\
                m\\boldsymbol{c}^T & m
            \\end{bmatrix}

        Parameters
        ----------
        J : np.ndarray (4, 4)
            The pseudo-inertia matrix.
        """
        assert J.shape == (4, 4)
        H = J[:3, :3]
        h = J[3, :3]
        mass = J[3, 3]
        return cls(mass=mass, h=h, H=H)

    @classmethod
    def from_point_masses(cls, masses, points):
        """Construct from a system of point masses.

        Parameters
        ----------
        masses : np.ndarray, shape (n,)
            The masses at each point.
        points : np.ndarray, shape (n, 3)
            The locations of each mass.
        """
        masses = np.array(masses)
        points = np.array(points)
        assert masses.shape[0] == points.shape[0]

        # homogeneous representation
        P = np.hstack((points, np.ones((points.shape[0], 1))))

        J = sum([m * np.outer(p, p) for m, p in zip(masses, P)])
        return cls.from_pim(J)

    @classmethod
    def random(cls, rng=None):
        """Generate a random set of fully physically consistent inertial parameters.

        Useful for testing purposes.
        """
        rng = np.random.default_rng(rng)
        mass = 0.1 + rng.random() * 0.9
        com = rng.random(3) - 0.5
        H = random_psd_matrix(n=3, rng=rng) + mass * np.outer(com, com)
        return cls(mass=mass, com=com, H=H)

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

    def consistent(self, eps=0):
        """Check if the inertial parameters are fully physically consistent.

        This means that there exists a physical rigid body that can have these
        parameters; a necessary and sufficient condition is that the
        pseudo-inertia matrix is positive semidefinite.

        Parameters
        ----------
        eps : float
            The parameters will be considered consistent if all of the
            eigenvalues of the pseudo-inertia matrix are greater than or equal
            to ``eps``.

        Returns
        -------
        : bool
            ``True`` if the parameters are consistent, ``False`` otherwise.
        """
        return np.min(np.linalg.eigvals(self.J)) >= eps

    def transform(self, rotation=None, translation=None):
        if rotation is None:
            rotation = np.eye(3)
        if translation is None:
            translation = np.zeros(3)

        # this is equivalent to computing T @ J @ T.T, where T is homeogeneous
        # transformation matrix representing the transform and J is the
        # pseudo-inertia matrix
        com = rotation @ self.com + translation
        H = rotation @ self.Hc @ rotation.T + self.mass * np.outer(com, com)
        return InertialParameters(mass=self.mass, com=com, H=H)

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
        return M @ A + skew6(V) @ M @ V
