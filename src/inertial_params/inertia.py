import numpy as np

import inertial_params as util


def pseudo_inertia_matrix(m, c, H):
    """Construct the pseudo-inertia matrix."""
    h = m * c
    J = np.zeros((4, 4))
    J[:3, :3] = H
    J[:3, 3] = h
    J[3, :3] = h
    J[3, 3] = m
    return J


def cuboid_inertia_matrix(mass, half_extents):
    """Inertia matrix for a rectangular cuboid with side_lengths in (x, y, z)
    dimensions."""
    lx, ly, lz = 2 * np.array(half_extents)
    xx = ly**2 + lz**2
    yy = lx**2 + lz**2
    zz = lx**2 + ly**2
    return mass * np.diag([xx, yy, zz]) / 12.0


def hollow_sphere_inertia_matrix(mass, radius):
    """Inertia matrix for a hollow sphere."""
    return mass * radius**2 * 2 / 3 * np.eye(3)


def solid_sphere_inertia_matrix(mass, radius):
    """Inertia matrix for a hollow sphere."""
    return mass * radius**2 * 2 / 5 * np.eye(3)


def point_mass_system_inertia(masses, points):
    """Inertia matrix corresponding to a finite set of point masses."""
    H = np.zeros((3, 3))
    for m, p in zip(masses, points):
        H += m * np.outer(p, p)
    return H, H2I(H)

def point_mass_system_h(masses, points):
    return np.sum(masses[:, None] * points, axis=0)

def point_mass_system_com(masses, points):
    """Center of mass of a finite set of point masses."""
    return point_mass_system_h(masses, points) / np.sum(masses)


def H2I(H):
    return np.trace(H) * np.eye(3) - H


def I2H(I):
    return 0.5 * np.trace(I) * np.eye(3) - I


def body_regressor(V, A):
    """Compute regressor matrix Y given body frame velocity V and acceleration A.

    The regressor maps the inertial parameters to the body inertial wrench: w = Yθ.
    """
    return util.lift6(A) + util.skew6(V) @ util.lift6(V)


class RigidBody:
    """Inertial parameters of a rigid body.

    Parameters
    ----------
    mass : float
        Mass of the body.
    com : iterable
        Center of mass of the body w.r.t. to some reference point O.
    I : np.ndarray
        3x3 inertia matrix about w.r.t. O
    """

    def __init__(self, mass, h, H, tol=1e-8):
        self.mass = mass
        self.h = h
        self.H = H

        assert mass >= -tol, f"Mass must be non-negative but is {mass}."

        min_H_λ = np.min(np.linalg.eigvals(self.H))
        assert min_H_λ >= -tol, f"H must be p.s.d. but min eigenval is {min_H_λ}"

    @property
    def com(self):
        return self.h / self.mass

    @property
    def θ(self):
        return np.concatenate([[self.mass], self.h, util.vech(self.I)])

    @property
    def I(self):
        return H2I(self.H)

    @property
    def J(self):
        J = np.zeros((4, 4))
        J[:3, :3] = self.H
        J[:3, 3] = self.h
        J[3, :3] = self.h
        J[3, 3] = self.mass
        return J

    @property
    def M(self):
        S = util.skew3(self.h)
        return np.block([[self.mass * np.eye(3), -S], [S, self.I]])

    @classmethod
    def from_vector(cls, θ):
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
        H = J[:3, :3]
        h = J[3, :3]
        mass = J[3, 3]
        return cls(mass=mass, h=h, H=H)

    @classmethod
    def from_point_masses(cls, masses, points):
        mass = sum(masses)
        h = point_mass_system_h(masses, points)
        H = point_mass_system_inertia(masses, points)[0]
        return cls(mass=mass, h=h, H=H)

    @classmethod
    def translate_from_com(cls, mass, h, Hc):
        H = Hc + np.outer(h, h) / mass
        return cls(mass=mass, h=h, H=H)

    @classmethod
    def from_mcI(cls, mass, com, I):
        h = com / mass
        H = I2H(I)
        return cls(mass=mass, h=h, H=H)

    def __add__(self, other):
        return RigidBody(
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