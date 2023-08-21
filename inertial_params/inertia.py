import numpy as np


def pseudo_inertia_matrix(m, c, H):
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


def H2I(H):
    return np.trace(H) * np.eye(3) - H


def I2H(I):
    return 0.5 * np.trace(I) * np.eye(3) - I
