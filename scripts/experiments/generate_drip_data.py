#!/usr/bin/env python3
"""Compare the moment SDP constraints with specialized custom constraints."""
import argparse
import datetime
import time

import numpy as np
import cvxpy as cp
import rigeo as rg
from scipy.spatial.transform import Rotation
import tqdm


def setup_drip_problem(mass, drip_constraints, **kwargs):
    J = cp.Variable((4, 4), PSD=True)
    D = cp.Parameter((4, 4), symmetric=True)

    objective = cp.Maximize(cp.trace(D @ J))
    constraints = [J[3, 3] == mass] + drip_constraints(J, **kwargs)
    problem = cp.Problem(objective, constraints)
    return problem, D


def random_optimal_values(shape, n, verbose=False):
    """Compare optimal extreme values."""
    rng = np.random.default_rng(0)

    mass = 1

    problem_moment_d2, D_moment_d2 = setup_drip_problem(
        mass, shape.moment_sdp_constraints, d=2
    )
    problem_moment_d3, D_moment_d3 = setup_drip_problem(
        mass, shape.moment_sdp_constraints, d=3
    )
    problem_custom, D_custom = setup_drip_problem(
        mass, shape.moment_custom_vertex_constraints
    )

    moment_d2_values = []
    moment_d3_values = []
    custom_values = []

    moment_d2_times = []
    moment_d3_times = []
    custom_times = []

    for i in tqdm.trange(n):
        Dvec = rng.uniform(low=-1, high=1, size=10)
        D = rg.unvech(Dvec)

        D_moment_d2.value = D
        D_moment_d3.value = D
        D_custom.value = D

        t0 = time.time()
        problem_moment_d2.solve(solver=cp.MOSEK)
        t1 = time.time()
        assert problem_moment_d2.status == "optimal"
        moment_d2_times.append(t1 - t0)
        moment_d2_values.append(problem_moment_d2.value)

        t0 = time.time()
        problem_moment_d3.solve(solver=cp.MOSEK)
        t1 = time.time()
        assert problem_moment_d3.status == "optimal"
        moment_d3_times.append(t1 - t0)
        moment_d3_values.append(problem_moment_d3.value)

        t0 = time.time()
        problem_custom.solve(solver=cp.MOSEK)
        t1 = time.time()
        assert problem_custom.status == "optimal"
        custom_times.append(t1 - t0)
        custom_values.append(problem_custom.value)

    return {
        "moment_d2_values": np.array(moment_d2_values),
        "moment_d3_values": np.array(moment_d3_values),
        "custom_values": np.array(custom_values),
        "moment_d2_times": np.array(moment_d2_times),
        "moment_d3_times": np.array(moment_d3_times),
        "custom_times": np.array(custom_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "shape",
        choices=["box", "cylinder"],
        help="Shape to use for the problem.",
    )
    parser.add_argument("-n", type=int, default=1000, help="Number of examples to generate.")
    args = parser.parse_args()

    if args.shape == "box":
        half_extents = np.array([0.5, 1, 1.5])
        shape = rg.Box(half_extents)
    elif args.shape == "cylinder":
        shape = rg.Cylinder(radius=0.5, length=1)
    else:
        raise ValueError("Invalid shape.")

    # NOTE: we could also transform the shape
    # R = Rotation.random(random_state=rng).as_matrix()
    # t = rng.uniform(low=-1, high=1, size=3)
    # shape = shape.transform(rotation=R, translation=t)

    data = random_optimal_values(shape, n=args.n)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile = f"{args.shape}_drip_data_{timestamp}.npz"
    np.savez(outfile, **data)
    print(f"Saved data to {outfile}")


if __name__ == "__main__":
    main()
