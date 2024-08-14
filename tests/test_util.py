import numpy as np
import rigeo as rg


def test_vech():
    A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    a = rg.vech(A)
    assert np.allclose(a, [1, 2, 3, 4, 5, 6])
    assert np.allclose(rg.unvech(a), A)

    rng = np.random.default_rng(0)
    for i in range(100):
        A = rg.random_psd_matrix(i + 1, rng=rng)
        a = rg.vech(A)
        assert np.allclose(rg.unvech(a), A)


def test_skew3():
    rng = np.random.default_rng(0)

    for _ in range(100):
        a = rng.uniform(-1, 1, size=3)
        b = rng.uniform(-1, 1, size=3)
        assert np.allclose(rg.skew3(a) @ b, np.cross(a, b))


def test_lift3():
    rng = np.random.default_rng(0)

    for _ in range(100):
        A = rng.uniform(-1, 1, size=(3, 3))
        A = 0.5 * (A + A.T)  # make symmetric
        x = rng.uniform(-1, 1, size=3)
        assert np.allclose(rg.lift3(x) @ rg.vech(A), A @ x)


def test_lift6():
    rng = np.random.default_rng(0)

    for _ in range(100):
        P = rg.InertialParameters.random(rng=rng)
        V = rng.uniform(-1, 1, size=6)
        assert np.allclose(rg.lift6(V) @ P.vec, P.M @ V)
