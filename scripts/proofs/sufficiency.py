import numpy as np
import cvxpy as cp

import inertial_params as ip

import IPython


def check_polyhedron_realizable(vertices, m, c, H):
    n = vertices.shape[0]
    h = m * c

    Vs = np.array([np.outer(v, v) for v in vertices])

    masses = cp.Variable(n)
    objective = cp.Minimize([0])
    constraints = [
        masses >= 0,
        cp.sum(masses) == m,
        masses.T @ vertices == h,
        H << cp.sum([m * V for m, V in zip(masses, Vs)]),
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    if problem.status == "optimal":
        # check
        try:
            assert np.isclose(np.sum(masses.value), m)
            assert np.allclose(masses.value @ vertices, h)
        except Exception as e:
            print("assertion failed!")
            IPython.embed()
        H_max = np.sum([m * V for m, V, in zip(masses.value, Vs)], axis=0)
        e, _ = np.linalg.eig(H_max - H)
        # assert np.min(e) >= -1e-8
        print("Inertial parameters are physically realizable.")
        return True, masses.value
    elif problem.status == "infeasible":
        print("Inertial parameters are infeasible.")
        return False, None
    return False, None


def max_point_in_direction(V, u):
    n = V.shape[0]
    d = cp.Variable(1)
    w = cp.Variable(n)

    objective = cp.Maximize(d)
    constraints = [
        d * u == V.T @ w,
        w >= 0,
        cp.sum(w) == 1,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    point = d.value * u
    return point, d.value


def main():
    np.random.seed(0)

    h = 0.5
    half_extents = h * np.ones(3)
    box = ip.Box(half_extents)
    vertices = box.vertices
    n = vertices.shape[0]

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    com = np.sum(vertices, axis=0) / 4
    vertices = vertices - com

    # points = box.grid(n=10)
    # Ps = np.array([np.outer(p, p) for p in points])

    Q = ip.cube_inscribed_ellipsoid(h).Q

    N = 1000
    for i in range(N):
        mass = 1.0
        # c = box.random_points()
        c = np.zeros(3)
        h = mass * c

        # second term ensures this satisfies Π >> 0
        Hc = 0.1 * mass * ip.random_psd_matrix((3, 3))
        H = Hc + mass * np.outer(c, c)
        params = ip.RigidBody(mass=mass, h=h, H=H)

        realizable, m0 = check_polyhedron_realizable(vertices, mass, c, H)
        if realizable:
            # points = np.vstack((vertices, c))
            points = vertices
            # points = np.vstack((vertices, box.random_points(shape=500)))
            # points = grid
            Ps = np.array([np.outer(p, p) for p in points])

            # λ, U = np.linalg.eig(Hc)
            #
            # d_max = np.zeros(3)
            # d_min = np.zeros(3)
            # for i in range(3):
            #     u = U[:, i]
            #     d_max[i] = max_point_in_direction(vertices, u)[1]
            #     d_min[i] = -max_point_in_direction(vertices, -u)[1]
            #
            # m2_2 = λ / (d_max * (d_max - d_min))
            # m2_1 = -m2_2 * d_max / d_min
            # m2 = np.concatenate((m2_1, m2_2))

            # # TODO in the non-symmetric case this needs to be more clever
            # # m2 = np.concatenate((m1, m1)) / 2
            # p1 = np.vstack(((d_min * U).T, (d_max * U).T))
            # Ps = np.array([np.outer(p, p) for p in p1])
            # H1 = sum([m * P for m, P in zip(m2, Ps)])
            # # eigenvecs are cols of U
            # IPython.embed()
            # return

            # check ellipsoid feasibility
            if np.trace(Q @ params.J) < 0:
                print("NOT realizable on inscribed ellipsoid")
            else:
                print("realizable on inscribed ellipsoid")

            m_opt = cp.Variable(points.shape[0])
            H_opt = cp.sum([m * P for m, P in zip(m_opt, Ps)])

            objective = cp.Minimize(cp.sum(m_opt))
            constraints = [
                m_opt >= 0,
                # cp.sum(m_opt) == mass,
                # m_opt.T @ points == h,
                # H_opt == H,
                cp.sum(m_opt) <= mass,
                H_opt == H,
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS)
            print(problem.status)
            if problem.status == "optimal":
                pass
            if problem.status == "infeasible":
                IPython.embed()
                return


main()
