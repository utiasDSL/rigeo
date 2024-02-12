"""Compute min and max values for H matrix entries."""
import numpy as np
import cvxpy as cp

import rigeo as rg

import IPython


def main():
    np.set_printoptions(precision=6, suppress=True)

    box = rg.Box(half_extents=[0.05, 0.05, 0.2], center=[0, 0, 0.2])
    com_bound = rg.Box(half_extents=[0.04, 0.04, 0.1], center=[0, 0, 0.2])
    mass_min = 0.9
    mass_max = 1.1

    ell = box.minimum_bounding_ellipsoid()

    Vs = [np.outer(v, v) for v in box.vertices]

    J = cp.Variable((4, 4), PSD=True)
    h = J[:3, 3]
    m = J[3, 3]
    constraints = (
        [
            m >= mass_min,
            m <= mass_max,
        ]
        + com_bound.must_contain(h, scale=m)
        + [cp.trace(E.Q @ J) >= 0 for E in box.as_ellipsoidal_intersection()]
    )

    for i in range(3):
        for j in range(i, 3):
            objective = cp.Maximize(J[i, j])
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)
            Hij_max = objective.value

            objective = cp.Minimize(J[i, j])
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)
            Hij_min = objective.value

            print(f"H[{i}, {j}]: min={Hij_min}, max={Hij_max}")


if __name__ == "__main__":
    main()
