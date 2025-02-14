"""Compare the moment SDP constraints with specialized box constraints."""
import datetime
import time

import numpy as np
import cvxpy as cp
import rigeo as rg
import tqdm

import IPython


N = 10000


def random_optimal_values():
    """Compare optimal extreme values."""
    rng = np.random.default_rng(0)

    mass = 1
    d = 2
    half_extents = np.array([0.5, 1, 1.5])
    box = rg.Box(half_extents)

    J_moment = cp.Variable((4, 4), PSD=True)
    D_moment = cp.Parameter((4, 4), symmetric=True)
    objective_moment = cp.Maximize(cp.trace(D @ J))

    # need to constrain the mass or the problem is unbounded
    constraints_moment = [J_moment[3, 3] == mass] + box.moment_sdp_constraints(
        J_moment, d=d
    )
    problem_moment = cp.Problem(objective_moment, constraints_moment)

    J_vertex = cp.Variable((4, 4), PSD=True)
    D_vertex = cp.Parameter((4, 4), symmetric=True)
    objective_vertex = cp.Maximize(cp.trace(D_vertex @ J))
    constraints_vertex = [
        J_vertex[3, 3] == mass
    ] + box.moment_box_vertex_constraints(J_vertex)
    problem_vertex = cp.Problem(objective_vertex, constraints_vertex)

    moment_values = []
    moment_times = []
    vertex_values = []
    vertex_times = []

    for i in tqdm.trange(N):
        Dvec = rng.uniform(low=-1, high=1, size=10)
        D = rg.unvech(Dvec)
        D_vertex.value = D
        D_moment.value = D

        try:
            t0 = time.time()
            problem_moment.solve(solver=cp.MOSEK)
            t1 = time.time()
            moment_times.append(t1 - t0)
            moment_values.append(problem_moment.value)
        except cp.error.SolverError:
            print("moment problem failed to solve - skipping")
            continue

        t0 = time.time()
        problem_vertex.solve(solver=cp.MOSEK)
        t1 = time.time()
        vertex_times.append(t1 - t0)
        vertex_values.append(problem_vertex.value)

        # print(f"moment time = {moment_time}")
        # print(f"vertex time = {vertex_time}")
        # print(problem_moment.value)
        # print(problem_vertex.value)

        if problem_vertex.value > problem_moment.value + 1e-3:
            print("vertex constraints not as tight!")
            IPython.embed()

    return {
        "moment_values": np.array(moment_values),
        "moment_times": np.array(moment_times),
        "vertex_values": np.array(vertex_values),
        "vertex_times": np.array(vertex_times),
    }


def main():
    data = random_optimal_values()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile = f"moment_vs_cuboid_{timestamp}.npz"
    np.savez(outfile, **data)
    print(f"Saved data to {outfile}")


if __name__ == "__main__":
    main()
