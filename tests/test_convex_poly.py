import numpy as np

import inertial_params as ip


def polyhedra_same(poly1, poly2):
    """Check if two polyhedra are the same."""
    return poly1.contains_polyhedron(poly2) and poly2.contains_polyhedron(poly1)


def test_aabb():
    box = ip.Box(half_extents=[0.5, 0.5, 0.5])
    poly = ip.ConvexPolyhedron.from_vertices(box.vertices)
    aabb = poly.aabb()
    assert np.allclose(aabb.center, box.center)
    assert np.allclose(aabb.half_extents, box.half_extents)
    assert np.allclose(poly.vertices, box.vertices)


def test_contains():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    nv = vertices.shape[0]
    poly = ip.ConvexPolyhedron.from_vertices(vertices)
    assert np.all(poly.contains(vertices))

    # generate some random points that are contained in the polyhedron
    np.random.seed(0)
    points = poly.random_points(10)
    assert np.all(poly.contains(points))

    # points outside the polyhedron
    points = np.array([[-0.1, -0.1, -0.1], [1.1, 0, 0], [0.5, 0.5, 0.5]])
    assert np.all(np.logical_not(poly.contains(points)))


def test_intersection():
    box1 = ip.Box(half_extents=np.ones(3))
    intersection = box1.intersect(box1)

    # intersection with self is self
    assert polyhedra_same(intersection, box1)

    box2 = ip.Box(half_extents=np.ones(3), center=1.9 * np.ones(3))
    intersection = box1.intersect(box2)
    box_expected = ip.Box(half_extents=0.05 * np.ones(3), center=0.95 * np.ones(3))
    assert polyhedra_same(intersection, box_expected)

    # order of intersection doesn't matter
    assert polyhedra_same(intersection, box2.intersect(box1))

    assert polyhedra_same(intersection, box1.intersect(intersection))

    # no overlap
    box3 = ip.Box(half_extents=np.ones(3), center=[2.1, 0, 0])
    assert box1.intersect(box3) is None

    # face contact
    box4 = ip.Box(half_extents=np.ones(3), center=[1.9, 1.9, 2])
    intersection = box1.intersect(box4)
    expected = ip.ConvexPolyhedron.from_vertices(
        [[1, 1, 1], [1, 0.9, 1], [0.9, 0.9, 1], [0.9, 1, 1]]
    )
    assert polyhedra_same(intersection, expected)


def test_grid():
    pass
