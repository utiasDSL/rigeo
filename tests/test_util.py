import numpy as np
import rigeo as rg


def test_vech():
    A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    a = rg.vech(A)
    assert np.allclose(a, [1, 2, 3, 4, 5, 6])
    assert np.allclose(rg.unvech(a), A)

    # test with k != 0
    assert np.allclose(rg.vech(A, k=1), [2, 3, 5])

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


def test_triangular_numbers():
    assert not rg.is_triangular_number(0)[0]
    assert not rg.is_triangular_number(-1)[0]

    n = 1
    x = n * (n + 1) // 2
    for i in range(1000):
        if i == x:
            # we expect the number to be triangular
            res, m = rg.is_triangular_number(i)
            assert res
            assert n == m

            # increment to the next triangular number
            n += 1
            x = n * (n + 1) // 2
        else:
            assert not rg.is_triangular_number(i)[0]


def test_box_vertices():
    half_extents = [1, 2, 3]

    x, y, z = half_extents
    v1 = np.array(
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

    v2 = rg.box_vertices(half_extents)
    assert np.allclose(v1, v2)
