#!/usr/bin/env python3
"""Compare the moment SDP constraints with specialized box constraints."""
import datetime
import time

import numpy as np
import cvxpy as cp
import rigeo as rg
from scipy.spatial.transform import Rotation
import tqdm

import IPython

N = 1000


def setup_drip_problem(mass, drip_constraints, **kwargs):
    J = cp.Variable((4, 4), PSD=True)
    D = cp.Parameter((4, 4), symmetric=True)

    objective = cp.Maximize(cp.trace(D @ J))
    constraints = [J[3, 3] == mass] + drip_constraints(J, **kwargs)
    problem = cp.Problem(objective, constraints)
    return problem, D


def random_optimal_values(verbose=False):
    """Compare optimal extreme values."""
    rng = np.random.default_rng(0)

    mass = 1
    half_extents = np.array([0.5, 1, 1.5])
    box = rg.Box(half_extents)

    # R = Rotation.random(random_state=rng).as_matrix()
    # t = rng.uniform(low=-1, high=1, size=3)
    # box = box.transform(rotation=R, translation=t)

    problem_moment_d2, D_moment_d2 = setup_drip_problem(
        mass, box.moment_sdp_constraints, d=2
    )
    problem_moment_d3, D_moment_d3 = setup_drip_problem(
        mass, box.moment_sdp_constraints, d=3
    )
    problem_box, D_box = setup_drip_problem(
        mass, box.moment_box_vertex_constraints
    )

    moment_d2_values = []
    moment_d3_values = []
    box_values = []

    moment_d2_times = []
    moment_d3_times = []
    box_times = []

    for i in tqdm.trange(N):
        Dvec = rng.uniform(low=-1, high=1, size=10)
        D = rg.unvech(Dvec)

        D_moment_d2.value = D
        D_moment_d3.value = D
        D_box.value = D

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
        problem_box.solve(solver=cp.MOSEK)
        t1 = time.time()
        assert problem_box.status == "optimal"
        box_times.append(t1 - t0)
        box_values.append(problem_box.value)

        if verbose:
            print(f"d2  = {moment_d2_values[-1]}")
            print(f"d3  = {moment_d3_values[-1]}")
            print(f"box = {box_values[-1]}")

            print(f"d2  = {moment_d2_times[-1]}")
            print(f"d3  = {moment_d3_times[-1]}")
            print(f"box = {box_times[-1]}\n")

    return {
        "moment_d2_values": np.array(moment_d2_values),
        "moment_d3_values": np.array(moment_d3_values),
        "box_values": np.array(box_values),
        "moment_d2_times": np.array(moment_d2_times),
        "moment_d3_times": np.array(moment_d3_times),
        "box_times": np.array(box_times),
    }


def main():
    data = random_optimal_values()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile = f"box_drip_data_{timestamp}.npz"
    np.savez(outfile, **data)
    print(f"Saved data to {outfile}")


if __name__ == "__main__":
    main()
