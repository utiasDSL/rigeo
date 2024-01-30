import numpy as np
import cvxpy as cp
from spatialmath.base import rotx, roty, rotz
import inertial_params as ip


def test_ellipsoid_sphere():
    ell = ip.Ellipsoid.sphere(radius=0.5)
    assert ell.contains([0.4, 0, 0])
    assert not ell.contains([0.6, 0, 0])
    assert ell.rank == 3
    assert not ell.degenerate()

    ell2 = ip.Ellipsoid.from_Ab(A=ell.A, b=ell.b)
    assert np.allclose(ell.Einv, ell2.Einv)
    assert np.allclose(ell.center, ell2.center)

    ell3 = ip.Ellipsoid.from_Q(Q=ell.Q)
    assert np.allclose(ell.Einv, ell3.Einv)
    assert np.allclose(ell.center, ell3.center)


def test_ellipsoid_hypersphere():
    ell = ip.Ellipsoid.sphere(radius=0.5, dim=4)
    assert ell.contains([0.4, 0, 0, 0])
    assert not ell.contains([0.6, 0, 0, 0])
    assert ell.rank == 4
    assert not ell.degenerate()

    ell2 = ip.Ellipsoid.from_Ab(A=ell.A, b=ell.b)
    assert np.allclose(ell.Einv, ell2.Einv)
    assert np.allclose(ell.center, ell2.center)

    ell3 = ip.Ellipsoid.from_Q(Q=ell.Q)
    assert np.allclose(ell.Einv, ell3.Einv)
    assert np.allclose(ell.center, ell3.center)


def test_cube_bounding_ellipsoid():
    h = 0.5
    half_lengths = h * np.ones(3)
    points = ip.Box(half_lengths).vertices
    ell = ip.minimum_bounding_ellipsoid(points)
    elld = ip.cube_bounding_ellipsoid(h)
    assert np.allclose(ell.Q, elld.Q)


def test_cube_bounding_ellipsoid_translated():
    h = 0.5
    offset = np.array([1, 1, 0])

    half_lengths = h * np.ones(3)
    points = ip.Box(half_lengths).vertices
    points += offset

    ell = ip.minimum_bounding_ellipsoid(points)
    elld = ip.cube_bounding_ellipsoid(h).transform(translation=offset)
    assert np.allclose(ell.Q, elld.Q)


def test_cube_bounding_ellipsoid_rotated():
    h = 0.5
    C = rotx(np.pi / 2) @ roty(np.pi / 4)

    half_lengths = h * np.ones(3)
    points = ip.Box(half_lengths).vertices
    points = (C @ points.T).T

    ell = ip.minimum_bounding_ellipsoid(points)
    elld = ip.cube_bounding_ellipsoid(h).transform(rotation=C)
    assert np.allclose(ell.Q, elld.Q)


def test_bounding_ellipoid_4d():
    np.random.seed(0)

    dim = 4
    points = np.random.random((20, dim))
    ell = ip.minimum_bounding_ellipsoid(points)
    for x in points:
        assert ell.contains(x)


def test_bounding_ellipsoid_degenerate():
    points = np.array([[0.5, 0, 0], [-0.5, 0, 0]])
    ell = ip.minimum_bounding_ellipsoid(points)
    assert ell.rank == 1
    assert ell.degenerate()
    for x in points:
        assert ell.contains(x)


def test_cube_inscribed_ellipsoid():
    h = 0.5
    half_lengths = h * np.ones(3)
    vertices = ip.Box(half_lengths).vertices
    ell = ip.maximum_inscribed_ellipsoid(vertices)
    elld = ip.cube_inscribed_ellipsoid(h)
    assert np.allclose(ell.Q, elld.Q)


def test_inscribed_sphere():
    box = ip.Box(half_extents=[1, 2, 3])
    ell = box.maximum_inscribed_ellipsoid(sphere=True)
    assert np.allclose(ell.Einv, np.eye(3))


def test_inscribed_ellipsoid_4d():
    np.random.seed(0)

    dim = 4
    points = np.random.random((20, dim))
    vertices = ip.convex_hull(points)
    # A, b = ip.polyhedron_span_to_face_form(vertices)
    ell = ip.maximum_inscribed_ellipsoid(vertices)

    # check that all the vertices are on the border or outside of the
    # ellipsoid
    for x in vertices:
        assert (x - ell.center).T @ ell.Einv @ (x - ell.center) >= 1


def test_inscribed_ellipsoid_degenerate():
    vertices = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    ell = ip.maximum_inscribed_ellipsoid(vertices)
    assert ell.rank == 2

    # check that all the vertices are on the border or outside of the
    # ellipsoid
    for x in vertices:
        assert (x - ell.center).T @ ell.Einv @ (x - ell.center) >= 1


def test_ellipsoid_must_contain():
    ell = ip.Ellipsoid.sphere(radius=1)
    point = cp.Variable(3)
    objective = cp.Maximize(point[0])
    constraints = ell.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 1.0)
