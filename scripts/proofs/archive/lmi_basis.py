"""This script is designed to test if we can always A = weighted sum of outer
products for A << B, with B a weighted sum of the same outer products.

In general this is not possible (see notes for a counter-example), but it is
not clear what happens when there is redundancy in the set of vectors.
"""
import numpy as np
import cvxpy as cp

import inertial_params as ip

import IPython


def find_coefficients(H, mhat, Vs, tol=1e-8):
    n = mhat.shape[0]
    ms = cp.Variable(n)

    E = tol * np.eye(3)
    H_opt = cp.sum([m * V for m, V in zip(ms, Vs)])
    H_max = H + E
    H_min = H - E

    objective = cp.Minimize(cp.sum(ms))
    # constraints = [ms >= 0, ms <= mhat, H_opt >> H_min, H_opt << H_max]
    constraints = [ms >= 0, ms <= mhat, H_opt == H]
    problem = cp.Problem(objective, constraints)

    # NOTE: SCS seems to work but MOSEK does not
    problem.solve(solver=cp.SCS, verbose=False)

    if problem.status != "optimal":
        print("problem infeasible!")
        IPython.embed()
        return
    else:
        assert np.allclose(H_opt.value, H)
        print("optimal")
    return ms.value


def main():
    np.random.seed(0)

    half_extents = 0.5 * np.ones(3)
    vertices = ip.AxisAlignedBox(half_extents).vertices
    n = vertices.shape[0]
    Vs = np.array([np.outer(v, v) for v in vertices])
    vs = np.array([ip.vech(V) for V in Vs])

    # NOTE messing around trying to find a counter-example
    # H = sum([0.125 * np.outer(v, v) for v in vertices])
    # vs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # c = sum([v / 4 for v in vs])
    # Hc = sum([np.outer(v - c, v - c) / 4 for v in vs])
    # es, us = np.linalg.eig(Hc)
    # H = sum([np.outer(v - c, v - c) / 3 for v in vs])

    IPython.embed()
    return

    N = 1000
    for i in range(N):
        C = ip.random_psd_matrix((3, 3))
        mhat = np.random.random(n)
        Hhat = np.sum(mhat[:, None, None] * Vs, axis=0)

        # 1. find an H << Hhat
        H = cp.Variable((3, 3), PSD=True)
        objective = cp.Maximize(cp.trace(C @ H))
        constraints = [H << Hhat]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)

        # 2. see if it can be expressed as sum of Vs
        find_coefficients(H.value, mhat, Vs)


if __name__ == "__main__":
    main()
