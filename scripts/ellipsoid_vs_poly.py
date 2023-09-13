"""This script compares the existing ellipsoidal check and the proposed
polyhedral check for feasible inertial parameters.

The idea is that if the ellipsoidal check says some params are feasible but the
polyhedral check says they are not, then the polyhedral check is tighter (thus
better). In particular, this is principled since there is a proposed proof that
the polyhedral check is necessary for feasibility.
"""
import numpy as np
import cvxpy as cp

import inertial_params as ip

import IPython


def check_feasible_params_ellipsoid(Q, m, c, H):
    J = ip.pseudo_inertia_matrix(m, c, H)
    return np.trace(Q @ J) >= 0


def check_feasible_params_polyhedron(vertices, m, c, H):
    n = vertices.shape[0]
    h = m * c
    Vs = np.array([np.outer(v, v) for v in vertices])

    # find a set of masses
    masses = cp.Variable(n)
    objective = cp.Minimize([0])
    constraints = [
        cp.sum(masses) == m,
        masses.T @ vertices == h,
        H << cp.sum([m * V for m, V in zip(masses, Vs)]),
        masses >= 0,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    print(f"status = {problem.status}")
    return problem.status == "optimal"


def main():
    h = 0.5
    # half_extents = h * np.ones(3)
    half_extents = np.array([0.25, 0.5, 1.0])
    vertices = ip.cuboid_vertices(half_extents)
    Q = ip.minimum_bounding_ellipsoid(vertices).Q

    N = 1000
    for i in range(N):
        m = 1
        c = ip.random_point_in_box(half_extents)

        # second term ensures this satisfies J >> 0
        Hc = 0.1 * m * ip.random_psd_matrix((3, 3))
        H = Hc + m * np.outer(c, c)

        # check that params are physically-realizable *at all*
        J = ip.pseudo_inertia_matrix(m, c, H)
        eigs = np.linalg.eigvals(J)
        if np.min(eigs) < 0:
            print("params not feasible")
            continue

        # check feasibility w.r.t. shape of the body
        f1 = check_feasible_params_ellipsoid(Q, m, c, H)
        f2 = check_feasible_params_polyhedron(vertices, m, c, H)
        # if f1 != f2:
        #     print(f"f1 = {f1}\nf2 = {f2}")
        #     IPython.embed()
        #     return
        if f2 and not f1:
            print(f"f1 = {f1}\nf2 = {f2}")
            IPython.embed()
            return
        else:
            print("same")


main()
