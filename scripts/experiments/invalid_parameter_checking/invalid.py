import numpy as np
import cvxpy as cp
import rigeo as rg

import IPython

N = 1000
SOLVER = cp.MOSEK


def necessary():
    rng = np.random.default_rng(0)

    mass = 1
    box = rg.Box.cube(half_extent=0.5)
    ell = box.mbe()

    J = cp.Parameter((4, 4), PSD=True)
    objective = cp.Minimize(0)
    constraints_moment_d2 = box.moment_sdp_constraints(J, d=2)
    constraints_moment_d3 = box.moment_sdp_constraints(J, d=3)
    constraints_ellipsoid = ell.moment_constraints(J)

    problem_moment_d2 = cp.Problem(objective, constraints_moment_d2)
    problem_moment_d3 = cp.Problem(objective, constraints_moment_d3)
    problem_ellipsoid = cp.Problem(objective, constraints_ellipsoid)

    n_d2 = 0
    n_d3 = 0
    n_ell = 0

    for i in range(N):
        print(i)

        c = box.random_points(rng=rng)
        Hc = 0.1 * rg.random_psd_matrix(3, rng=rng)
        H = Hc + mass * np.outer(c, c)
        params = rg.InertialParameters(mass=mass, com=c, H=H)

        J.value = params.J

        # check feasibility with various conditions
        try:
            problem_moment_d2.solve(solver=SOLVER)
        except cp.error.SolverError:
            print("moment d=2: failed to solve")
            IPython.embed()

        try:
            problem_moment_d3.solve(solver=SOLVER)
        except cp.error.SolverError:
            print("moment d=3: failed to solve")
            IPython.embed()

        problem_ellipsoid.solve(solver=SOLVER)

        if problem_moment_d2.status == cp.OPTIMAL:
            n_d2 += 1

        if problem_moment_d3.status == cp.OPTIMAL:
            n_d3 += 1

        if problem_ellipsoid.status == cp.OPTIMAL:
            n_ell += 1

    print(f"moment d=2: {n_d2}/{N}")
    print(f"moment d=3: {n_d3}/{N}")
    print(f"ellipsoid:  {n_ell}/{N}")



def sufficient(n_points):
    rng = np.random.default_rng(0)

    mass = 1
    box = rg.Box.cube(half_extent=0.5)

    # various discretization resolutions
    grid1 = box.vertices
    grid2 = box.grid(3)
    grid3 = box.grid(5)
    grid4 = box.grid(11)

    # θd = cp.Parameter(10)
    # θ = cp.Variable(10)
    # J = rg.pim_must_equal_vec(θ)
    J = cp.Variable((4, 4), PSD=True)
    Jd = cp.Parameter((4, 4), PSD=True)

    # objective = cp.Minimize(cp.norm2(θd - θ))
    objective = cp.Minimize(cp.norm(Jd - J, "fro"))

    def _disc_problem(grid):
        μs = cp.Variable(grid.shape[0], nonneg=True)
        Ps = [np.append(p, 1) for p in grid]
        constraints = [J == cp.sum([μ * np.outer(p, p) for μ, p in zip(μs, Ps)])]
        return cp.Problem(objective, constraints)

    problem1 = _disc_problem(grid1)
    problem2 = _disc_problem(grid2)
    problem3 = _disc_problem(grid3)
    problem4 = _disc_problem(grid4)

    # params = rg.InertialParameters.from_point_masses(
    #     masses=[1], points=[[0.45, 0.45, 0.45]]
    # )
    # Jd.value = params.J
    # problem4.solve(solver=SOLVER)
    # assert problem4.status == cp.OPTIMAL
    # print(f"problem 4 = {problem4.value}")
    # IPython.embed()
    # return

    # TODO we may also want to try using the geodesic distance
    # TODO maybe this is always fine? can you prove the worst case?

    for i in range(N):
        points = box.random_points(shape=n_points, rng=rng)
        points = np.atleast_2d(points)  # for n_points = 1
        masses = rng.uniform(0, 1, size=n_points)
        masses = mass * masses / np.sum(masses)
        params = rg.InertialParameters.from_point_masses(
            masses=masses, points=points
        )

        # θd.value = params.vec
        Jd.value = params.J

        problem1.solve(solver=SOLVER)
        assert problem1.status == cp.OPTIMAL
        print(f"problem 1 = {problem1.value}")

        problem2.solve(solver=SOLVER)
        assert problem2.status == cp.OPTIMAL
        print(f"problem 2 = {problem2.value}")

        problem3.solve(solver=SOLVER)
        assert problem3.status == cp.OPTIMAL
        print(f"problem 3 = {problem3.value}")

        problem4.solve(solver=SOLVER)
        assert problem4.status == cp.OPTIMAL
        print(f"problem 4 = {problem4.value}")
        # if problem4.value > 0.0045:
        #     IPython.embed()



# necessary()
sufficient(n_points=1)
