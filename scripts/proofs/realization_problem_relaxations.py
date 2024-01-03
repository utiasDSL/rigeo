"""Relaxations/variations of the problem of finding a valid density to realize
a given set of inertial parameters."""
import numpy as np
import cvxpy as cp

import inertial_params as ip

import IPython


# NOTE: this is a relaxed version of the problem of finding realizing point
# masses (we have H >> h @ h.T / m rather than ==)
# So far I cannot find an objective that forces the relaxation to be tight in
# general
def relaxed_realization_problem(vertices, params):
    """Solve relaxed version of the problem of finding a realizing density for
    given inertial parameters.

    NOTE: so far I have not been able to find an objective function that makes
    the relaxation tight in general.
    """
    Vs = vertices - params.com
    Hc = params.Hc
    m = params.mass

    A, b = ip.polyhedron_span_to_face_form(Vs)
    # n = len(vertices)
    # eigvals, eigvecs = np.linalg.eig(Hc)
    # D = np.vstack((eigvecs, -eigvecs))
    n = vertices.shape[0]

    masses = cp.Variable(n)
    Hs = [cp.Variable((3, 3), PSD=True) for _ in range(n)]
    hs = [cp.Variable(3) for _ in range(n)]
    ts = cp.Variable(n)

    # objective = cp.Minimize(cp.sum([-h.T @ v for h, v in zip(hs, D)]))
    # objective = cp.Minimize([0])
    # objective = cp.Maximize(cp.sum(ts))
    objective = cp.Minimize(
        cp.sum([cp.quad_form(h - m * v, np.eye(3)) for h, v, m in zip(hs, Vs, masses)])
    )
    constraints = (
        [
            cp.sum(masses) == m,
            masses >= 0,
            ts >= 0,
            cp.sum(hs) == np.zeros(3),
            Hc == cp.sum(Hs),
        ]
        + [A @ hs[i] <= masses[i] * b for i in range(n)]
        + [ip.schur(H, h, m) >> 0 for t, H, h, m in zip(ts, Hs, hs, masses)]
    )
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    if problem.status != "optimal":
        print("problem not optimal")
        IPython.embed()
        raise ValueError()

    H_opt = sum([np.outer(h.value, h.value) / m.value for m, h in zip(masses, hs)])
    hs_opt = [h.value for h in hs]
    err = np.linalg.norm(Hc - H_opt)
    if err > 1e-4:
        print(f"error = {err}")
        IPython.embed()
        # raise ValueError()


def vech(X):
    return np.array([X[0, 0], X[0, 1], X[0, 2], X[1, 1], X[1, 2], X[2, 2]])


def vech_eq_con(X, x):
    return [X[0, :] == x[:3], X[1, :] == x[3:5], X[2, 2] == x[5]]


def find_realizing_point_masses2(vertices, params):
    vs = vertices - params.com
    Hc = params.Hc
    m = params.mass

    n = vertices.shape[0]
    Vs = [np.outer(v, v) for v in np.vstack((vs, np.zeros((1, 3))))]
    # Vs = [np.outer(v, v) for v in vs]
    # Vs_vec = np.array([vech(V) for V in Vs])
    # A, b = ip.polyhedron_span_to_face_form(Vs_vec)

    masses = cp.Variable(n)
    h = cp.Variable(3)

    # TODO this is still just looking at the vertices, which is not fully
    # general
    objective = cp.Minimize(cp.quad_form(h, np.eye(3), assume_PSD=True))
    constraints = [
        cp.sum(masses) == m,
        masses >= 0,
        h == cp.sum([m * v for m, v in zip(masses, vs)]),
        Hc << cp.sum([m * V for m, V in zip(masses, Vs)]),
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    if problem.status != "optimal":
        print(f"problem is {problem.status}")
        IPython.embed()
        raise ValueError()
    h_opt = sum([m.value * v for m, v in zip(masses, vs)])
    H_opt = sum([m.value * V for m, V in zip(masses, Vs)])
    print(f"error = {np.linalg.norm(Hc - H_opt)}")


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    N = 1000  # number of trials
    n = 20  # number of points per trial

    # box
    half_extents = 0.5 * np.ones(3)
    box = ip.AxisAlignedBox(half_extents)

    # sample a random polyhedron with masses
    # masses = np.random.random(n) + 1e-8  # avoid any possibility of zero mass
    # masses /= np.sum(masses)
    # points = box.random_points(n)
    # poly_params = ip.RigidBody.from_point_masses(masses=masses, points=points)
    # poly = ip.ConvexPolyhedron.from_convex_hull_of(points)

    for i in range(N):
        print(i)
        masses = np.random.random(n) + 1e-8  # avoid any possibility of zero mass
        masses /= np.sum(masses)
        points = box.random_points(n)
        params = ip.RigidBody.from_point_masses(masses=masses, points=points)
        vertices = ip.convex_hull(points)

        relaxed_realization_problem(vertices, params)


def main2():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    # box
    half_extents = 0.5 * np.ones(3)
    box = ip.AxisAlignedBox(half_extents)

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    masses = np.ones(3) / 3
    params1 = ip.RigidBody.from_point_masses(masses=masses, points=vertices)

    p1 = np.array([0.5, 0.5, 0])
    # p2 = params1.com - (p1 - params1.com)
    p2 = np.array([0, 0, 0])
    params2 = ip.RigidBody.from_point_masses(
        masses=np.array([2 / 3, 1 / 3]), points=np.vstack((p1, p2))
    )

    for a in np.linspace(0, 1, 11):
        p = np.array([1 - a, a])
        P = np.outer(p, p)
        # print(np.linalg.norm(p))
        print(np.linalg.eigvals(P))
        V = (1 - a) * np.outer([1, 0], [1, 0]) + a * np.outer([0, 0.5], [0, 0.5])
        print(np.trace(V))

    IPython.embed()

    # sample a random polyhedron with masses
    # masses = np.random.random(n) + 1e-8  # avoid any possibility of zero mass
    # masses /= np.sum(masses)
    # points = box.random_points(n)
    # poly_params = ip.RigidBody.from_point_masses(masses=masses, points=points)
    # poly = ip.ConvexPolyhedron.from_convex_hull_of(points)


if __name__ == "__main__":
    main2()
