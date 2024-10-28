import numpy as np
import cvxpy as cp
from scipy.spatial.transform import Rotation

import rigeo as rg


def test_cube_at_origin_feasibility():
    rng = np.random.default_rng(0)

    N = 100  # number of trials
    n = 10  # number of point masses per trial

    box = rg.Box(half_extents=[0.5, 0.5, 0.5])

    # set up the feasibility problem
    J = cp.Parameter((4, 4), PSD=True)
    problem = box.moment_sdp_feasibility_problem(J)

    for i in range(N):
        points = box.random_points(n, rng=rng)
        masses = rng.random(n)
        params = rg.InertialParameters.from_point_masses(masses=masses, points=points)

        # solve the problem
        J.value = params.J
        problem.solve(solver=cp.MOSEK)
        assert problem.status == "optimal"

    points = np.array([-box.half_extents, box.half_extents])
    masses = [0.5, 0.5]
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    J.value = params.J
    problem.solve(solver=cp.MOSEK)
    assert problem.status == "optimal"

    masses = np.ones(8)
    params = rg.InertialParameters.from_point_masses(masses=masses, points=box.vertices)
    J.value = params.J
    problem.solve(solver=cp.MOSEK)
    assert problem.status == "optimal"

    # infeasible case
    points = np.array([-box.half_extents, 1.1 * box.half_extents])
    masses = [0.5, 0.5]
    params = rg.InertialParameters.from_point_masses(masses=masses, points=points)
    J.value = params.J
    problem.solve(solver=cp.MOSEK)
    assert not problem.status == "optimal"


def test_box_offset_from_origin_feasibility():
    rng = np.random.default_rng(0)

    N = 100  # number of trials
    n = 10  # number of point masses per trial

    box = rg.Box(half_extents=[1.0, 0.5, 0.1], center=[1, 1, 0])

    # set up the feasibility problem
    J = cp.Parameter((4, 4), PSD=True)
    problem = box.moment_sdp_feasibility_problem(J)

    for i in range(N):
        points = box.random_points(n, rng=rng)
        masses = rng.random(n)
        params = rg.InertialParameters.from_point_masses(masses=masses, points=points)

        # solve the problem
        J.value = params.J
        problem.solve(solver=cp.MOSEK)
        assert problem.status == "optimal"


def test_box_transformed_feasibility():
    rng = np.random.default_rng(0)

    N = 100  # number of trials
    n = 10  # number of point masses per trial

    for i in range(N):
        # random box
        half_extents = rng.uniform(low=0.5, high=1.5, size=3)
        center = rng.uniform(low=-2, high=2, size=3)
        C = Rotation.random(random_state=rng).as_matrix()
        box = rg.Box(half_extents=half_extents, center=center, rotation=C)

        # random parameters
        points = box.random_points(n, rng=rng)
        masses = rng.random(n)
        params = rg.InertialParameters.from_point_masses(masses=masses, points=points)

        # solve the feasibility problem
        J = cp.Parameter((4, 4), PSD=True, value=params.J)
        problem = box.moment_sdp_feasibility_problem(J)
        problem.solve(solver=cp.MOSEK)

        assert problem.status == "optimal"


def test_cube_moment_sdp_constraints_J():
    box = rg.Box(half_extents=[0.5, 0.5, 0.5])

    # convert to a general convex polyhedron
    poly = rg.ConvexPolyhedron.from_vertices(box.vertices)

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]

    objective = cp.Maximize(J[0, 0])

    # need a mass constraint to bound the problem
    constraints = box.moment_sdp_constraints(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5**2)

    # objective should be the same if we use the general convex polyhedron
    # formulation
    constraints = poly.moment_sdp_constraints(J) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5**2)


def test_cube_moment_sdp_constraints_vec():
    box = rg.Box(half_extents=[0.5, 0.5, 0.5])
    poly = rg.ConvexPolyhedron.from_vertices(box.vertices)

    θ = cp.Variable(10)
    m = θ[0]

    objective = cp.Maximize(θ[4])

    # need a mass constraint to bound the problem
    constraints = box.moment_sdp_constraints(θ) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5)

    # objective should be the same if we use the general convex polyhedron
    # formulation
    constraints = poly.moment_sdp_constraints(θ) + [m <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    assert np.isclose(objective.value, 0.5)
