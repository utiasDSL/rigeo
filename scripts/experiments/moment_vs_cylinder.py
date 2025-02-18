"""Compare the moment SDP constraints with specialized box constraints."""
import datetime
import time
from functools import partial

import numpy as np
import cvxpy as cp
import rigeo as rg
import tqdm

import IPython

# TODO do bounding ellipsoid as well

N = 10


def setup_drip_problem(mass, drip_constraints):
    J = cp.Variable((4, 4), PSD=True)
    D = cp.Parameter((4, 4), symmetric=True)

    c = J[:3, 3] / mass  # CoM
    m = J[3, 3]  # mass

    objective = cp.Maximize(cp.trace(D @ J))
    constraints = [J[3, 3] == mass] + drip_constraints(J)
    problem = cp.Problem(objective, constraints)
    return problem, D


def random_optimal_values():
    """Compare optimal extreme values."""
    rng = np.random.default_rng(0)

    mass = 1
    cylinder = rg.Cylinder(radius=0.5, length=1)

    problem_moment_d2, D_moment_d2 = setup_drip_problem(
        mass, partial(cylinder.moment_sdp_constraints, d=2)
    )
    problem_moment_d3, D_moment_d3 = setup_drip_problem(
        mass, partial(cylinder.moment_sdp_constraints, d=3)
    )
    problem_box, D_box = setup_drip_problem(
        mass, cylinder.moment_cylinder_vertex_constraints
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

        # try:
        #     t0 = time.time()
        #     problem_moment.solve(solver=cp.MOSEK)
        #     t1 = time.time()
        #     moment_times.append(t1 - t0)
        #     moment_values.append(problem_moment.value)
        # except cp.error.SolverError:
        #     print("moment problem failed to solve - skipping")
        #     continue

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

        print(f"d2  = {problem_moment_d2.value}")
        print(f"d3  = {problem_moment_d3.value}")
        print(f"box = {problem_box.value}")

        print(f"d2  = {moment_d2_times[-1]}")
        print(f"d3  = {moment_d3_times[-1]}")
        print(f"box = {box_times[-1]}\n")

        # if problem_box.value > problem_moment.value + 1e-3:
        #     print("box constraints not as tight!")
        #     IPython.embed()
    raise ValueError("stop")

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
    outfile = f"moment_vs_box_{timestamp}.npz"
    np.savez(outfile, **data)
    print(f"Saved data to {outfile}")


if __name__ == "__main__":
    main()
