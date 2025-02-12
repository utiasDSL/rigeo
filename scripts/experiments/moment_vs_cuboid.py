"""Compare the moment SDP constraints with my custom box constraints."""
import time

import numpy as np
import cvxpy as cp
import rigeo as rg

import IPython


def feasibility():
    """Compare feasibility of the two problem approaches."""
    rng = np.random.default_rng(0)

    mass = 1
    d = 3
    half_extents = np.array([0.5, 1, 1.5])
    box = rg.Box(half_extents)

    J = cp.Parameter((4, 4), PSD=True)
    objective = cp.Minimize(0)
    constraints_moment = box.moment_sdp_constraints(J, d=d)
    constraints_vertex = box.moment_box_vertex_constraints(J)

    problem_moment = cp.Problem(objective, constraints_moment)
    problem_vertex = cp.Problem(objective, constraints_vertex)

    # check basic feasibility
    for i in range(1000):
        print(i)

        c = box.random_points(rng=rng)
        Hc = 0.1 * rg.random_psd_matrix(3, rng=rng)
        H = Hc + mass * np.outer(c, c)
        params = rg.InertialParameters(mass=mass, com=c, H=H)

        J.value = params.J

        try:
            problem_moment.solve(solver=cp.MOSEK)
        except cp.error.SolverError:
            print("moment problem failed to solve - skipping")
            continue

        problem_vertex.solve(solver=cp.MOSEK)

        moment_feas = problem_moment.status == "optimal"
        vertex_feas = problem_vertex.status == "optimal"
        if vertex_feas != moment_feas:
            print("different results!")
            if vertex_feas and not moment_feas:
                IPython.embed()
        else:
            print(f"feasible = {vertex_feas}")

        # if loc_feas:
        #     print(f"rank = {np.linalg.matrix_rank(Md1_var.value)}")


def optimal_values():
    """Compare optimal extreme values."""
    rng = np.random.default_rng(0)

    mass = 1
    d = 2
    half_extents = np.array([0.5, 1, 1.5])
    box = rg.Box(half_extents)

    J = cp.Variable((4, 4), PSD=True)
    D = cp.Parameter((4, 4), symmetric=True)
    objective = cp.Maximize(cp.trace(D @ J))

    # need to constrain the mass or the problem is unbounded
    constraints = [J[3, 3] == mass]
    constraints_moment = constraints + box.moment_sdp_constraints(J, d=d)
    constraints_vertex = constraints + box.moment_box_vertex_constraints(J)

    problem_moment = cp.Problem(objective, constraints_moment)
    problem_vertex = cp.Problem(objective, constraints_vertex)

    # check basic feasibility
    for i in range(1000):
        print(i)

        Dvec = rng.uniform(low=-1, high=1, size=10)
        D.value = rg.unvech(Dvec)

        t0 = time.time()
        problem_vertex.solve(solver=cp.MOSEK)
        t1 = time.time()
        vertex_time = t1 - t0

        try:
            t0 = time.time()
            problem_moment.solve(solver=cp.MOSEK)
            t1 = time.time()
            moment_time = t1 - t0
        except cp.error.SolverError:
            print("moment problem failed to solve - skipping")
            continue

        print(f"moment time = {moment_time}")
        print(f"vertex time = {vertex_time}")

        # print(problem_moment.value)
        # print(problem_vertex.value)

        if problem_vertex.value > problem_moment.value + 1e-3:
            print("vertex constraints not as tight!")
            IPython.embed()
        elif not np.isclose(problem_vertex.value, problem_moment.value, atol=1e-3):
            print(f"diff = {problem_moment.value - problem_vertex.value}")


# feasibility()
optimal_values()
