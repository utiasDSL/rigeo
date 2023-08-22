import numpy as np


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
    return H, np.trace(H) * np.eye(3) - H


def point_mass_system_com(masses, points):
    """Center of mass of a finite set of point masses."""
    return np.sum(masses[:, None] * points, axis=0) / np.sum(masses)


def H2I(H):
    return np.trace(H) * np.eye(3) - H


def I2H(I):
    return 0.5 * np.trace(I) * np.eye(3) - I
