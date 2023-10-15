import numpy as np
from scipy.linalg import sqrtm


def random_psd_matrix(shape):
    A = 2 * np.random.random(shape) - 1
    return sqrtm(A.T @ A)


# def random_point_in_box(half_extents, n=1):
#     """Generate a set of random points in a box with given half_extents."""
#     assert n >= 1
#     d = half_extents.shape[0]
#     points = half_extents * (2 * np.random.random((n, d)) - 1)
#     if n == 1:
#         return points.flatten()
#     return points
