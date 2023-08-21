import numpy as np


def cuboid_vertices(half_extents):
    """Vertices of a cuboid with given half extents."""
    x, y, z = half_extents
    return np.array(
        [
            [x, y, z],
            [x, y, -z],
            [x, -y, z],
            [x, -y, -z],
            [-x, y, z],
            [-x, y, -z],
            [-x, -y, z],
            [-x, -y, -z],
        ]
    )


def cube_bounding_ellipsoid(h):
    """Bounding ellipsoid (sphere) of a cube with half length h.

    Returns:
        4x4 matrix Q such that (x, 1).T @ Q @ (x, 1) >= 0 implies x is inside
        the ellipsoid.
    """
    r = np.linalg.norm([h, h, h])
    return np.diag(np.append(-np.ones(3) / r**2, 1))
