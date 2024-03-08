import numpy as np
import rigeo as rg


def test_vech():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a = rg.vech(A)
    assert np.allclose(a, [1, 2, 3, 5, 6, 9])


def test_skew3():
    np.random.seed(0)

    for _ in range(100):
        a = 2 * np.random.random(3) - 1
        b = 2 * np.random.random(3) - 1
        assert np.allclose(rg.skew3(a) @ b, np.cross(a, b))


def test_lift3():
    np.random.seed(0)

    for _ in range(100):
        A = 2 * np.random.random((3, 3)) - 1
        A = 0.5 * (A + A.T)  # make symmetric
        x = 2 * np.random.random(3) - 1
        assert np.allclose(rg.lift3(x) @ rg.vech(A), A @ x)


def test_lift6():
    np.random.seed(0)

    for _ in range(100):
        P = rg.InertialParameters.random()
        V = 2 * np.random.random(6) - 1
        assert np.allclose(rg.lift6(V) @ P.vec, P.M @ V)
