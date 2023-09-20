import numpy as np
from spatialmath.base import rotx, roty, rotz
import inertial_params as ip


def test_ellipsoid_sphere():
    ell = ip.Ellipsoid.sphere(radius=0.5)
    assert ell.contains([0.4, 0, 0])
    assert not ell.contains([0.6, 0, 0])

    ell2 = ip.Ellipsoid.from_Ab(A=ell.A, b=ell.b)
    assert np.allclose(ell.Einv, ell2.Einv)
    assert np.allclose(ell.c, ell2.c)

    ell3 = ip.Ellipsoid.from_Q(Q=ell.Q)
    assert np.allclose(ell.Einv, ell3.Einv)
    assert np.allclose(ell.c, ell3.c)


def test_cube_bounding_ellipsoid():
    h = 0.5
    half_lengths = h * np.ones(3)
    points = ip.AxisAlignedBox(half_lengths).vertices
    ell = ip.minimum_bounding_ellipsoid(points)
    elld = ip.cube_bounding_ellipsoid(h)
    assert np.allclose(ell.Q, elld.Q)


def test_cube_bounding_ellipsoid_translated():
    h = 0.5
    offset = np.array([1, 1, 0])

    half_lengths = h * np.ones(3)
    points = ip.AxisAlignedBox(half_lengths).vertices
    points += offset

    ell = ip.minimum_bounding_ellipsoid(points)
    elld = ip.cube_bounding_ellipsoid(h).transform(r=offset)
    assert np.allclose(ell.Q, elld.Q)


def test_cube_bounding_ellipsoid_rotated():
    h = 0.5
    C = rotx(np.pi / 2) @ roty(np.pi / 4)

    half_lengths = h * np.ones(3)
    points = ip.AxisAlignedBox(half_lengths).vertices
    points = (C @ points.T).T

    ell = ip.minimum_bounding_ellipsoid(points)
    elld = ip.cube_bounding_ellipsoid(h).transform(C=C)
    assert np.allclose(ell.Q, elld.Q)


# def test_line_bounding_ellipsoid():
#     points = np.array([[0.5, 0, 0], [-0.5, 0, 0]])
#     # TODO we need to project points into the nullspace to avoid degenerate
#     # ellipsoids
#     Q = ip.minimum_bounding_ellipsoid(points)
#     print(Q)
