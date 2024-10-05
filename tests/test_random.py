import numpy as np
import pytest

import rigeo as rg


def test_random_weight_vectors():
    rng = np.random.default_rng(0)

    # single weight vector
    w = rg.random_weight_vectors(5, rng=rng)
    assert w.shape == (5,)
    assert np.all(w >= 0)
    assert np.isclose(np.sum(w), 1)

    # set of weight vectors
    w = rg.random_weight_vectors((10, 5), rng=rng)
    assert w.shape == (10, 5)
    assert np.all(w >= 0)
    assert np.allclose(np.sum(w, axis=1), 1)

    # even higher dimension
    w = rg.random_weight_vectors((10, 10, 5), rng=rng)
    assert w.shape == (10, 10, 5)
    assert np.all(w >= 0)
    assert np.allclose(np.sum(w, axis=2), 1)


def test_random_psd_matrix():
    rng = np.random.default_rng(0)

    for i in range(10):
        n = i + 2
        A = rg.random_psd_matrix(n, rng=rng)
        assert np.min(np.linalg.eigvals(A)) >= 0


def test_random_points_on_hypersphere():
    rng = np.random.default_rng(0)

    # one point
    point = rg.random_points_on_hypersphere(dim=2, rng=rng)
    assert point.shape == (3,)
    assert np.isclose(np.linalg.norm(point), 1.0)

    # multiple points
    points = rg.random_points_on_hypersphere(shape=10, dim=2, rng=rng)
    assert points.shape == (10, 3)
    assert np.allclose(np.linalg.norm(points, axis=-1), 1.0)

    # hypersphere
    points = rg.random_points_on_hypersphere(shape=10, dim=3, rng=rng)
    assert np.allclose(np.linalg.norm(points, axis=-1), 1.0)

    # grid of points
    points = rg.random_points_on_hypersphere(shape=(10, 10), dim=2, rng=rng)
    assert points.shape == (10, 10, 3)
    assert np.allclose(np.linalg.norm(points, axis=-1), 1.0)

    # grid with one dimension 1
    points = rg.random_points_on_hypersphere(shape=(10, 1), dim=2, rng=rng)
    assert points.shape == (10, 1, 3)
    assert np.allclose(np.linalg.norm(points, axis=-1), 1.0)


def test_random_points_in_ball():
    rng = np.random.default_rng(0)

    # one point
    point = rg.random_points_in_ball(dim=3, rng=rng)
    assert point.shape == (3,)
    assert np.linalg.norm(point) <= 1.0

    # multiple points
    points = rg.random_points_in_ball(shape=10, dim=3, rng=rng)
    assert points.shape == (10, 3)
    assert np.all(np.linalg.norm(points, axis=-1) <= 1.0)

    # hyperball
    points = rg.random_points_in_ball(shape=10, dim=4, rng=rng)
    assert points.shape == (10, 4)
    assert np.all(np.linalg.norm(points, axis=-1) <= 1.0)

    # grid of points
    points = rg.random_points_in_ball(shape=(10, 10), dim=3, rng=rng)
    assert points.shape == (10, 10, 3)
    assert np.all(np.linalg.norm(points, axis=-1) <= 1.0)

    # grid with one dimension 1
    points = rg.random_points_in_ball(shape=(10, 1), dim=3, rng=rng)
    assert points.shape == (10, 1, 3)
    assert np.all(np.linalg.norm(points, axis=-1) <= 1.0)

    # test that the sampling is uniform by seeing if the number of points that
    # fall in an inscribed box is proportional to its relative volume
    n = 10000
    points = rg.random_points_in_ball(shape=n, dim=3, rng=rng)
    inbox = rg.Box.cube(half_extent=1 / np.sqrt(3))
    n_box = np.sum(inbox.contains(points))

    vol_ball = 4 * np.pi / 3
    vol_box = inbox.volume

    # accuracy can be increased by increasing n
    assert np.isclose(n_box / n, vol_box / vol_ball, rtol=0, atol=0.01)


def test_rejection_sample():
    rng = np.random.default_rng(0)

    # test a single actual shape
    box_actual = rg.Box.cube(half_extent=0.5)
    box_bounding = rg.Box.cube(half_extent=1)

    points = rg.rejection_sample(
        actual_shapes=[box_actual],
        bounding_shape=box_bounding,
        sample_shape=100,
        rng=rng,
    )
    assert box_actual.contains(points).all()

    # multiple actual shapes
    box_actual = rg.Box.cube(half_extent=0.5, center=[2, 0, 0])
    ell_actual = rg.Ellipsoid.sphere(radius=0.5, center=[2.5, 0, 0])
    box_bounding = rg.Box.cube(half_extent=1, center=[2, 0, 0])

    # don't include zero to ensure the generated array of points is being
    # populated properly
    assert not box_actual.contains([0, 0, 0])
    assert not ell_actual.contains([0, 0, 0])

    points = rg.rejection_sample(
        actual_shapes=[box_actual, ell_actual],
        bounding_shape=box_bounding,
        sample_shape=100,
        rng=rng,
    )
    assert np.all(box_actual.contains(points) | ell_actual.contains(points))

    # when the actual shapes is not actually contained in the bounding shape,
    # eventually the algorithm errors out
    box_actual = rg.Box.cube(half_extent=0.5)
    box_bounding = rg.Box.cube(half_extent=1, center=[2, 0, 0])
    with pytest.raises(ValueError):
        rg.rejection_sample(
            actual_shapes=[box_actual],
            bounding_shape=box_bounding,
            sample_shape=100,
            rng=rng,
            max_tries=100,
        )
