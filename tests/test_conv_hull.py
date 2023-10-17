import numpy as np
import inertial_params as ip


def allclose_unordered(A, B):
    assert A.shape == B.shape
    n = A.shape[0]
    B_checked = np.zeros(n, dtype=bool)
    for i in range(n):
        a = A[i, :]
        residuals = np.linalg.norm(B - a, axis=1)

        # False where residual = 0, True otherwise
        mask = ~np.isclose(residuals, 0)

        # False where residual = 0 AND B has not been checked yet
        test = np.logical_or(mask, B_checked)

        # check to see if we have any cases where the test passes
        idx = np.argmin(test)
        if not test[idx]:
            B_checked[idx] = True
        else:
            return False
    return True


def test_allclose_unordered():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

    # reorder A
    B = A[[1, 0, 2], :]
    assert allclose_unordered(A, B)

    C = np.copy(A)
    C[0, 0] = 0
    assert not allclose_unordered(A, C)


def test_conv_hull_cube():
    half_extents = 0.5 * np.ones(3)
    points = ip.AxisAlignedBox(half_extents).vertices
    vertices = ip.convex_hull(points)
    assert allclose_unordered(vertices, points)

    # generate some random points inside the hull
    np.random.seed(0)
    extras = ip.AxisAlignedBox(half_extents).random_points(10)
    points_extra = np.vstack((points, extras))
    vertices = ip.convex_hull(points_extra)
    assert allclose_unordered(vertices, points)


def test_conv_hull_degenerate():
    # just a square, with no z variation
    points = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    vertices = ip.convex_hull(points)
    assert allclose_unordered(vertices, points)

    # generate some random points inside the hull
    np.random.seed(0)
    extras = 2 * np.random.random((10, 2)) - 1
    extras = np.hstack((extras, np.zeros((10, 1))))
    points_extra = np.vstack((points, extras))
    vertices = ip.convex_hull(points_extra)
    assert allclose_unordered(vertices, points)
