import numpy as np

import rigeo as rg


def test_conv_hull_cube():
    half_extents = 0.5 * np.ones(3)
    points = rg.Box(half_extents).vertices
    vertices = rg.convex_hull(points)
    assert rg.allclose_unordered(vertices, points)

    # generate some random points inside the hull
    rng = np.random.default_rng(0)
    extras = rg.Box(half_extents).random_points(10, rng=rng)
    points_extra = np.vstack((points, extras))
    vertices = rg.convex_hull(points_extra)
    assert rg.allclose_unordered(vertices, points)


def test_conv_hull_degenerate():
    # just a square, with no z variation
    points = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    vertices = rg.convex_hull(points)
    assert rg.allclose_unordered(vertices, points)

    # generate some random points inside the hull
    rng = np.random.default_rng(0)
    extras = rng.uniform(low=-1, high=1, size=(10, 2))
    extras = np.hstack((extras, np.zeros((10, 1))))
    points_extra = np.vstack((points, extras))
    vertices = rg.convex_hull(points_extra)
    assert rg.allclose_unordered(vertices, points)
