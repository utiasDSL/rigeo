import numpy as np
import cvxpy as cp
from spatialmath.base import rotx, roty, rotz

import pytest

import rigeo as rg


def test_aabb():
    box = rg.Box(half_extents=[0.5, 0.5, 0.5])
    poly = rg.ConvexPolyhedron.from_vertices(box.vertices)
    aabb = poly.aabb()
    assert np.allclose(aabb.center, box.center)
    assert np.allclose(aabb.half_extents, box.half_extents)
    assert np.allclose(poly.vertices, box.vertices)


def test_contains():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    nv = vertices.shape[0]
    poly = rg.ConvexPolyhedron.from_vertices(vertices)
    assert np.all(poly.contains(vertices))

    # generate some random points that are contained in the polyhedron
    rng = np.random.default_rng(0)
    points = poly.random_points(10, rng=rng)
    assert np.all(poly.contains(points))

    # points outside the polyhedron
    points = np.array([[-0.1, -0.1, -0.1], [1.1, 0, 0], [0.5, 0.5, 0.5]])
    assert np.all(np.logical_not(poly.contains(points)))


def test_intersection():
    box1 = rg.Box(half_extents=np.ones(3))
    intersection = box1.intersect(box1)

    # intersection with self is self
    assert intersection.is_same(box1)

    box2 = rg.Box(half_extents=np.ones(3), center=1.9 * np.ones(3))
    intersection = box1.intersect(box2)
    box_expected = rg.Box(half_extents=0.05 * np.ones(3), center=0.95 * np.ones(3))
    assert intersection.is_same(box_expected)

    # order of intersection doesn't matter
    assert intersection.is_same(box2.intersect(box1))

    assert intersection.is_same(box1.intersect(intersection))

    # no overlap
    box3 = rg.Box(half_extents=np.ones(3), center=[2.1, 0, 0])
    assert box1.intersect(box3) is None

    # face contact
    box4 = rg.Box(half_extents=np.ones(3), center=[1.9, 1.9, 2])
    intersection = box1.intersect(box4)
    expected = rg.ConvexPolyhedron.from_vertices(
        [[1, 1, 1], [1, 0.9, 1], [0.9, 0.9, 1], [0.9, 1, 1]]
    )
    assert intersection.is_same(expected)


def test_degenerate():
    # here we mostly want to ensure the face form behaves as expected in the
    # dengenerate case
    poly = rg.ConvexPolyhedron.from_vertices([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert poly.A.shape == (5, 3)
    assert poly.b.shape == (5,)

    # ensure contains works as expected
    assert poly.contains([0.5, 0.5, 0])
    assert not poly.contains([0.5, 0.5, 0.1])
    assert not poly.contains([0.5, 0.5, -0.1])

    # ensure we recover the same vertices when going back to span form from
    # inequality-only face form
    poly2 = rg.ConvexPolyhedron(span_form=poly.face_form.to_span_form())
    assert poly2.is_same(poly)


def test_unbounded():
    # half space is unbounded
    A = np.array([[1, 0, 0]])
    b = np.array([0])
    with pytest.raises(ValueError):
        poly = rg.ConvexPolyhedron.from_halfspaces(A, b)


def test_grid():
    rng = np.random.default_rng(0)

    # random polyhedron
    vertices = rng.random((10, 3))
    poly = rg.ConvexPolyhedron.from_vertices(vertices, prune=True)
    grid = poly.grid(10)
    assert poly.contains(grid).all()


def test_must_contain():
    poly = rg.ConvexPolyhedron.from_vertices(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    point = cp.Variable(3)

    objective = cp.Maximize(point[0])
    constraints = poly.must_contain(point)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 1.0)


def test_random_points():
    rng = np.random.default_rng(0)

    # up-cast box to general poly so specialized random_points method for boxes
    # not used
    C = rotx(np.pi / 4) @ roty(np.pi / 6)
    box = rg.Box(half_extents=[2, 1, 0.5], center=[1, 0, 1], rotation=C).as_poly()

    # one point
    point = box.random_points(rng=rng)
    assert point.shape == (3,)
    assert box.contains(point)

    # multiple points
    points = box.random_points(shape=10, rng=rng)
    assert points.shape == (10, 3)
    assert box.contains(points).all()

    # grid of points
    points = box.random_points(shape=(10, 10), rng=rng)
    assert points.shape == (10, 10, 3)
    assert box.contains(points.reshape((100, 3))).all()

    # grid with one dimension 1
    points = box.random_points(shape=(10, 1), rng=rng)
    assert points.shape == (10, 1, 3)
    assert box.contains(points.reshape((10, 3))).all()
