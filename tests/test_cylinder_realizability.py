import numpy as np
import cvxpy as cp

import rigeo as rg


def test_cylinder_at_origin():
    rng = np.random.default_rng(0)

    N = 100  # number of trials
    n = 10  # number of point masses per trial

    cylinder = rg.Cylinder(length=1, radius=0.5)

    # set up the feasibility problem
    J = cp.Parameter((4, 4), PSD=True)
    problem = cylinder.moment_sdp_feasibility_problem(J)

    for i in range(N):
        points = cylinder.random_points(n, rng=rng)
        masses = rng.random(points.shape[0])
        params = rg.InertialParameters.from_point_masses(masses=masses, points=points)

        # solve the problem
        J.value = params.J
        problem.solve(solver=cp.MOSEK)
        assert problem.status == "optimal"

    points = np.array([[0, 0.5, 0.5], [0, -0.5, -0.5]])
    masses = [0.5, 0.5]
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    J.value = params.J
    problem.solve(solver=cp.MOSEK)
    assert problem.status == "optimal"

    # fmt: off
    points = 0.5 * np.array([
        [1., 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
        [1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]])
    # fmt: on
    masses = np.ones(8)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    J.value = params.J
    problem.solve(solver=cp.MOSEK)
    assert problem.status == "optimal"

    # infeasible cases
    points = 1.1 * np.array([[0, 0.5, 0.5], [0, -0.5, -0.5]])
    masses = [0.5, 0.5]
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    J.value = params.J
    problem.solve(solver=cp.MOSEK)
    assert not problem.status == "optimal"

    points = 0.6 * np.array([
        [1., 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
        [1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]])
    # fmt: on
    masses = np.ones(8)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    J.value = params.J
    problem.solve(solver=cp.MOSEK)
    assert not problem.status == "optimal"


def test_cylinder_moment_vertex_constraints():
    cylinder = rg.Cylinder(length=1, radius=0.5)

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]

    objective = cp.Maximize(J[0, 0])

    # need a mass constraint to bound the problem
    constraints = cylinder.moment_custom_vertex_constraints(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert problem.status == cp.OPTIMAL
    assert np.isclose(objective.value, 0.25)

    objective = cp.Maximize(J[2, 2])
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert problem.status == cp.OPTIMAL
    assert np.isclose(objective.value, 0.25)

    # now try offset from the origin
    cylinder = rg.Cylinder(length=1, radius=0.5, center=[1, 0, 0])

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]

    objective = cp.Maximize(J[0, 0])
    constraints = cylinder.moment_custom_vertex_constraints(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert problem.status == cp.OPTIMAL

    # optimum is obtained by putting the CoM at the maximum x-position
    assert np.isclose(objective.value, 1.5**2)
