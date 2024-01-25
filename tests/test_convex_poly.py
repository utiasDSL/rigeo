import numpy as np

import inertial_params as ip


def test_aabb():
    box = ip.AxisAlignedBox(half_extents=[0.5, 0.5, 0.5])
    poly = ip.ConvexPolyhedron(box.vertices)
    aabb = poly.aabb()
    assert np.allclose(aabb.center, box.center)
    assert np.allclose(aabb.half_extents, box.half_extents)
    assert np.allclose(poly.vertices, box.vertices)

def test_contains():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    nv = vertices.shape[0]
    poly = ip.ConvexPolyhedron(vertices)
    assert np.all(poly.contains(vertices))

    # generate some random points that are contained in the polyhedron
    np.random.seed(0)
    points = poly.random_points(10)
    assert np.all(poly.contains(points))

    # points outside the polyhedron
    points = np.array([[-0.1, -0.1, -0.1], [1.1, 0, 0], [0.5, 0.5, 0.5]])
    assert np.all(np.logical_not(poly.contains(points)))

    print(ip.random_weight_vectors(10))

def test_grid():
    pass
