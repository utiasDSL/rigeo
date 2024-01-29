import numpy as np
import cvxpy as cp
import cdd
from scipy.spatial import ConvexHull
import inertial_params as ip
# from inertial_params.geometry import _polyhedron_span_to_face_form, _polyhedron_face_to_span_form

import IPython

def convex_hull(points):
    n = points.shape[0]
    Smat = cdd.Matrix(np.hstack((np.ones((n, 1)), points)))
    Smat.rep_type = cdd.RepType.GENERATOR

    poly = cdd.Polyhedron(Smat)

    Fmat = poly.get_generators()
    F = np.array([Fmat[i] for i in range(Fmat.row_size)])
    if Fmat.row_size == 0:
        return None

    # we are assuming a closed polyhedron, so there are no rays
    t = F[:, 0]
    assert np.allclose(t, 1.0)

    # vertices
    V = F[:, 1:]
    return V


# vertices = ip.Box(half_extents=[0.5, 0.5, 0.5]).vertices
# hull = ConvexHull(vertices)
# A = hull.equations[:, :-1]
# b = hull.equations[:, -1]
# A, b = _polyhedron_span_to_face_form(vertices)
# V = _polyhedron_face_to_span_form(A, b)

# box1 = ip.Box(half_extents=[0.5, 0.5, 0.5])
# box2 = ip.Box(half_extents=[0.5, 0.5, 0.5], center=[1.1, 0, 0])

# A = np.vstack((box1.A, box2.A))
# b = np.concatenate((box1.b, box2.b))
# points = np.array([[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], [0, 0, 0]])
# V = convex_hull(points)


# TODO this does not handle degeneracy properly
poly = ip.ConvexPolyhedron(
    vertices=np.array([[1, 1, 1], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]])
)

IPython.embed()
