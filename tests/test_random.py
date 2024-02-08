import numpy as np

import inertial_params as ip


def test_random_weight_vectors():
    # single weight vector
    w = ip.random_weight_vectors(5)
    assert w.shape == (5,)
    assert np.all(w >= 0)
    assert np.isclose(np.sum(w), 1)

    # set of weight vectors
    w = ip.random_weight_vectors((10, 5))
    assert w.shape == (10, 5)
    assert np.all(w >= 0)
    assert np.allclose(np.sum(w, axis=1), 1)

    # even higher dimension
    w = ip.random_weight_vectors((10, 10, 5))
    assert w.shape == (10, 10, 5)
    assert np.all(w >= 0)
    assert np.allclose(np.sum(w, axis=2), 1)


def test_random_psd_matrix():
    np.random.seed(0)
    for i in range(10):
        n = i + 2
        A = ip.random_psd_matrix(n)
        assert np.min(np.linalg.eigvals(A)) >= 0


def test_random_points_on_hypersphere():
    np.random.seed(0)

    # one point
    point = ip.random_points_on_hypersphere(dim=2)
    assert point.shape == (3,)
    assert np.isclose(np.linalg.norm(point), 1.0)

    # multiple points
    points = ip.random_points_on_hypersphere(shape=10, dim=2)
    assert points.shape == (10, 3)
    assert np.allclose(np.linalg.norm(points, axis=-1), 1.0)

    # hypersphere
    points = ip.random_points_on_hypersphere(shape=10, dim=3)
    assert np.allclose(np.linalg.norm(points, axis=-1), 1.0)

    # grid of points
    points = ip.random_points_on_hypersphere(shape=(10, 10), dim=2)
    assert points.shape == (10, 10, 3)
    assert np.allclose(np.linalg.norm(points, axis=-1), 1.0)

    # grid with one dimension 1
    points = ip.random_points_on_hypersphere(shape=(10, 1), dim=2)
    assert points.shape == (10, 1, 3)
    assert np.allclose(np.linalg.norm(points, axis=-1), 1.0)


def test_random_points_in_ball():
    np.random.seed(0)

    # one point
    point = ip.random_points_in_ball(dim=3)
    assert point.shape == (3,)
    assert np.linalg.norm(point) <= 1.0

    # multiple points
    points = ip.random_points_in_ball(shape=10, dim=3)
    assert points.shape == (10, 3)
    assert np.all(np.linalg.norm(points, axis=-1) <= 1.0)

    # hyperball
    points = ip.random_points_in_ball(shape=10, dim=4)
    assert points.shape == (10, 4)
    assert np.all(np.linalg.norm(points, axis=-1) <= 1.0)

    # grid of points
    points = ip.random_points_in_ball(shape=(10, 10), dim=3)
    assert points.shape == (10, 10, 3)
    assert np.all(np.linalg.norm(points, axis=-1) <= 1.0)

    # grid with one dimension 1
    points = ip.random_points_in_ball(shape=(10, 1), dim=3)
    assert points.shape == (10, 1, 3)
    assert np.all(np.linalg.norm(points, axis=-1) <= 1.0)

    # test that the sampling is uniform by seeing if the number of points that
    # fall in an inscribed box is proportional to its relative volume
    n = 10000
    points = ip.random_points_in_ball(shape=n, dim=3)
    inbox = ip.Box.cube(half_extent=1 / np.sqrt(3))
    n_box = np.sum(inbox.contains(points))

    vol_ball = 4 * np.pi / 3
    vol_box = inbox.volume

    # accuracy can be increased by increasing n
    assert np.isclose(n_box / n, vol_box / vol_ball, rtol=0, atol=0.01)
