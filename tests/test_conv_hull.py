import numpy as np
import inertial_params as ip


def sort_canonical(A):
    """Helper to sort an nd-array into a canonical order, axis by axis.

    Note that this does not preserve any given axis; the order of every axis of
    elements may be changed, but no element value is lost.
    """
    B = np.copy(A)
    for i in range(B.ndim):
        B.sort(axis=-i - 1)
    return B


def test_conv_hull_cube():
    half_extents = 0.5 * np.ones(3)
    points = ip.AxisAlignedBox(half_extents).vertices
    vertices = ip.convex_hull(points)
    assert np.allclose(sort_canonical(vertices), sort_canonical(points))

    # generate some random points inside the hull
    np.random.seed(0)
    extras = ip.AxisAlignedBox(half_extents).random_points(10)
    points_extra = np.vstack((points, extras))
    vertices = ip.convex_hull(points_extra)
    assert np.allclose(sort_canonical(vertices), sort_canonical(points))


def test_conv_hull_degenerate():
    # just a square, with no z variation
    points = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    vertices = ip.convex_hull(points)
    assert np.allclose(sort_canonical(vertices), sort_canonical(points))

    # generate some random points inside the hull
    np.random.seed(0)
    extras = 2 * np.random.random((10, 2)) - 1
    extras = np.hstack((extras, np.zeros((10, 1))))
    points_extra = np.vstack((points, extras))
    vertices = ip.convex_hull(points_extra)
    assert np.allclose(sort_canonical(vertices), sort_canonical(points))
