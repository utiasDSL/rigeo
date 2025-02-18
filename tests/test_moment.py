import numpy as np
from scipy.linalg import hankel
import rigeo as rg


def test_polynomial_init():
    # two ways to construct the same polynomial
    p1 = rg.Polynomial({"000": 1, "100": -1, "010": -1})
    p2 = rg.Polynomial.affine(a=[1, 1, 0], b=1)
    assert p1.coefficients == p2.coefficients


def test_polynomial_degree():
    assert rg.Polynomial({"000": 1}).degree == 0
    assert rg.Polynomial({"100": 1}).degree == 1
    assert rg.Polynomial({"110": 1}).degree == 2
    assert rg.Polynomial({"002": 1}).degree == 2

    assert rg.Polynomial.ones(n=3, d=2).degree == 2


def test_polynomial_evaluate():
    p = rg.Polynomial({"100": 1, "001": -1})

    assert np.isclose(p.evaluate(x=[2, 0, 0]), 2)
    assert np.isclose(p.evaluate(x=[0, 0, 1]), -1)
    assert np.isclose(p.evaluate(x=[1, 0, 1]), 0)

    p = rg.Polynomial({"200": 1})

    assert np.isclose(p.evaluate(x=[2, 0, 0]), 4)


def test_polynomial_add():
    p1 = rg.Polynomial({"100": 1})
    p2 = rg.Polynomial({"200": 1})
    p = p1 + p2

    assert np.isclose(p.evaluate(x=[2, 0, 0]), 6)

    p1 = rg.Polynomial({"100": 1})
    p2 = rg.Polynomial({"100": -1})
    p = p1 + p2

    assert np.isclose(p.evaluate(x=[2, 0, 0]), 0)


def test_moment_matrix():
    M = rg.MomentMatrix(n=1, d=2)
    M_expected = np.array(
        [["0", "1", "2"], ["1", "2", "3"], ["2", "3", "4"]], dtype=object
    )
    for i in range(M_expected.shape[0]):
        for j in range(M_expected.shape[1]):
            assert rg.MomentIndex(M_expected[i, j]) == M.indices[i, j]
