import numpy as np

import rigeo as rg


def test_allclose_unordered():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

    # reorder A
    B = A[[1, 0, 2], :]
    assert rg.allclose_unordered(A, B)

    C = np.copy(A)
    C[0, 0] = 0
    assert not rg.allclose_unordered(A, C)

    # test with repeated row
    D = np.copy(A)
    D[1, :] = D[0, :]
    assert not rg.allclose_unordered(A, D)
    assert not rg.allclose_unordered(D, A)
    assert rg.allclose_unordered(D, D)
