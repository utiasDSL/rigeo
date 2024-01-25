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
