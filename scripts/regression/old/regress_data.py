"""Regress the inertial parameters from trajectory data using differently
constrained least squares problems and noise corruption."""
import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import pybullet_data
import pinocchio
import cvxpy as cp

import inertial_params as ip

import IPython


# these values are taken from the Robotiq FT-300 datasheet
WRENCH_STDEV = np.array([1.2, 1.2, 0.5, 0.02, 0.02, 0.03])

DISC_GRID_SIZE = 5
TRAIN_TEST_SPLIT = 0.5
TRAIN_WITH_NOISY_Y = False
USE_BOUNDING_BOX = False


def J_vec_constraint(J, θ):
    """Constraint to enforce consistency between J and θ representations."""
    H = J[:3, :3]
    I = cp.trace(H) * np.eye(3) - H
    return [
        J[3, 3] == θ[0],
        J[:3, 3] == θ[1:4],
        I[0, :3] == θ[4:7],
        I[1, 1:3] == θ[7:9],
        I[2, 2] == θ[9],
    ]


class DiscretizedIPIDProblem:
    def __init__(self, Ys, ws, ws_noise, cov):
        self.A = np.vstack(Ys)
        self.b = np.concatenate(ws + ws_noise)
        self.θ = cp.Variable(10)
        self.Jopt = cp.Variable((4, 4), PSD=True)

        cov_inv = np.linalg.inv(cov)
        self.W = np.kron(np.eye(Ys.shape[0]), cov_inv)

    def solve(self, points):
        masses = cp.Variable(points.shape[0])
        # objective = cp.Minimize(cp.sum_squares(self.A @ self.θ - self.b))
        objective = cp.Minimize(cp.quad_form(self.A @ self.θ - self.b, self.W))
        Ps = np.array([np.outer(p, p) for p in points])
        constraints = J_vec_constraint(self.Jopt, self.θ) + [
            masses >= 0,
            cp.sum(masses) == self.θ[0],
            masses.T @ points == self.θ[1:4],
            self.Jopt[:3, :3] == cp.sum([m * P for m, P in zip(masses, Ps)]),
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        # print(f"disc solve time = {problem.solver_stats.solve_time}")
        # print(f"disc value = {problem.value}")
        return ip.RigidBody.from_vector(self.θ.value)


class IPIDProblem:
    """Inertial parameter identification optimization problem."""

    def __init__(self, Ys, ws, ws_noise, cov):
        self.A = np.vstack(Ys)
        self.b = np.concatenate(ws + ws_noise)
        self.θ = cp.Variable(10)
        self.Jopt = cp.Variable((4, 4), PSD=True)

        cov_inv = np.linalg.inv(cov)
        self.W = np.kron(np.eye(Ys.shape[0]), cov_inv)

    def _solve(self, extra_constraints=None, name=None):
        # objective = cp.Minimize(cp.sum_squares(self.A @ self.θ - self.b))
        objective = cp.Minimize(cp.quad_form(self.A @ self.θ - self.b, self.W))

        constraints = J_vec_constraint(self.Jopt, self.θ)
        if extra_constraints is not None:
            constraints.extend(extra_constraints)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        # if name is not None:
        #     # print(f"{name} solve time = {problem.solver_stats.solve_time}")
        #     print(f"{name} value = {problem.value}")
        return ip.RigidBody.from_vector(self.θ.value)

    def solve_nominal(self):
        return self._solve()

    def solve_ellipsoid(self, ellipsoid):
        extra_constraints = [cp.trace(ellipsoid.Q @ self.Jopt) >= 0]
        return self._solve(extra_constraints)

    def solve_polyhedron(self, vertices):
        # TODO use a different reference point
        # center = np.mean(vertices, axis=0)
        # vertices = vertices - center
        # A, b = ip.polyhedron_span_to_face_form(vertices)

        nv = vertices.shape[0]
        mvs = cp.Variable(nv)
        Vs = np.array([np.outer(v, v) for v in vertices])
        extra_constraints = [
            mvs >= 0,
            cp.sum(mvs) == self.θ[0],
            mvs.T @ vertices == self.θ[1:4],
            self.Jopt[:3, :3] << cp.sum([m * V for m, V in zip(mvs, Vs)]),
            # A @ self.θ[1:4] <= self.θ[0] * b,
        ]

        # mass = self.θ[0].value
        # params_offset = ip.RigidBody(mass, mass * center, mass * np.outer(center, center))
        # params = self._solve(extra_constraints, name="poly")
        # return params + params_offset
        return self._solve(extra_constraints, name="poly")

    def solve_polyhedron2(self, vertices):
        nv = vertices.shape[0]
        mvs = cp.Variable(nv)
        αs = cp.Variable(nv)
        Vs = np.array([np.outer(v, v) for v in vertices])
        extra_constraints = [
            mvs >= 0,
            αs >= 0,
            αs <= mvs,
            cp.sum(mvs) == self.θ[0],
            mvs.T @ vertices == self.θ[1:4],
            self.Jopt[:3, :3] == cp.sum([m * V for m, V in zip(αs, Vs)]),
        ]
        return self._solve(extra_constraints, name="poly")

    def solve_polyhedron3(self, vertices):
        nv = vertices.shape[0]
        mvs = cp.Variable(nv)
        Vs = np.array([np.outer(v, v) for v in vertices])
        A, b = ip.polyhedron_span_to_face_form(vertices)

        extra_constraints = [
            mvs >= 0,
            cp.sum(mvs) == self.θ[0],
            mvs.T @ vertices == self.θ[1:4],
            # self.Jopt[:3, :3] == cp.sum([m * V for m, V in zip(mvs, Vs)]),
        ] + [
            cp.trace(self.Jopt[:3, :3] @ np.outer(a, a)) <= bi**2 * self.θ[0]
            for a, bi in zip(A, b)
        ]
        return self._solve(extra_constraints, name="poly")

    def solve_polyhedron4(self, vertices):
        nv = vertices.shape[0]
        mvs = cp.Variable(nv)
        Vs = np.array([np.outer(v, v) for v in vertices])
        A, b = ip.polyhedron_span_to_face_form(vertices)

        Hbar = cp.Variable((3, 3), PSD=True)
        Hc = cp.Variable((3, 3), PSD=True)
        H = self.Jopt[:3, :3]

        extra_constraints = [
            mvs >= 0,
            cp.sum(mvs) == self.θ[0],
            mvs.T @ vertices == self.θ[1:4],
            # Hc == H - Hbar,
            # schur(Hbar, self.θ[1:4], self.θ[0]) >> 0,
            # Hc << cp.sum([m * V for m, V in zip(mvs, Vs)]),
            H << cp.sum([m * V for m, V in zip(mvs, Vs)]),
        ]  # + [cp.trace(Hc @ np.outer(a, a)) <= bi**2 * self.θ[0] for a, bi in zip(A, b)]
        return self._solve(extra_constraints, name="poly")

    def solve_polyhedron5(self, vertices):
        # nv = vertices.shape[0]
        # mvs = cp.Variable(nv)
        # Vs = np.array([np.outer(v, v) for v in vertices])
        A, b = ip.polyhedron_span_to_face_form(vertices)

        Hbar = cp.Variable((3, 3), PSD=True)
        Hc = cp.Variable((3, 3), PSD=True)
        H = self.Jopt[:3, :3]

        extra_constraints = [
            self.θ[0] >= 0,
            A @ self.θ[1:4] <= self.θ[0] * b,
            Hc == H - Hbar,
            ip.schur(Hbar, self.θ[1:4], self.θ[0]) >> 0,
            H << cp.sum([m * V for m, V in zip(mvs, Vs)]),
        ] + [
            cp.trace(Hc @ np.outer(a, a)) <= bi**2 * self.θ[0] for a, bi in zip(A, b)
        ]
        return self._solve(extra_constraints, name="poly")

    def solve_both(self, vertices, ellipsoid):
        nv = vertices.shape[0]
        mvs = cp.Variable(nv)
        Vs = np.array([np.outer(v, v) for v in vertices])
        extra_constraints = [
            mvs >= 0,
            cp.sum(mvs) == self.θ[0],
            mvs.T @ vertices == self.θ[1:4],
            self.Jopt[:3, :3] << cp.sum([m * V for m, V in zip(mvs, Vs)]),
            cp.trace(ellipsoid.Q @ self.Jopt) >= 0,
        ]
        return self._solve(extra_constraints)


def validation_rmse(Ys, ws, θ):
    error = Ys @ θ - ws
    square = np.sum(error**2, axis=1)
    mean = np.mean(square)
    root = np.sqrt(mean)
    return root


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="File to load the data from.")
    args = parser.parse_args()

    with open(args.infile, "rb") as f:
        data = pickle.load(f)

    for i in range(data["num_obj"]):
        params = data["params"][i]
        vertices = data["vertices"][i]

        if USE_BOUNDING_BOX:
            box = ip.Box.from_points_to_bound(vertices)
            vertices = box.vertices
            grid = box.grid(n=DISC_GRID_SIZE)
        else:
            grid = ip.polyhedron_grid(vertices, n=DISC_GRID_SIZE)

        ellipsoid = ip.minimum_bounding_ellipsoid(vertices)

        Ys = np.array(data["obj_data"][i]["Ys"])
        Ys_noisy = np.array(data["obj_data"][i]["Ys_noisy"])
        ws = np.array(data["obj_data"][i]["ws"])
        n = Ys.shape[0]
        n_train = int(TRAIN_TEST_SPLIT * n)
        # n_test = n - n_train

        if TRAIN_WITH_NOISY_Y:
            Ys_train = Ys_noisy[:n_train]
        else:
            Ys_train = Ys[:n_train]
        ws_train = ws[:n_train]

        # generate random noise on the force measurements
        ws_noise = np.random.normal(scale=WRENCH_STDEV, size=ws_train.shape)

        # solve the problem with no noise
        prob_noiseless = IPIDProblem(Ys_train, ws_train, np.zeros_like(ws_noise), np.eye(6))
        params_noiseless = prob_noiseless.solve_nominal()

        # solve noisy problem with varying constraints
        cov = np.diag(WRENCH_STDEV**2)
        prob = IPIDProblem(Ys_train, ws_train, ws_noise, cov)
        params_nom = prob.solve_nominal()
        params_poly = prob.solve_polyhedron(vertices)
        params_ell = prob.solve_ellipsoid(ellipsoid)
        params_both = prob.solve_both(vertices, ellipsoid)

        params_grid = DiscretizedIPIDProblem(Ys_train, ws_train, ws_noise, cov).solve(grid)

        Ys_test = Ys[n_train:]
        ws_test = ws[n_train:]

        print(f"\nProblem {i + 1}")
        print(f"nv = {vertices.shape[0]}")
        print(f"ng = {grid.shape[0]}")
        print("\nθ error")
        print("-------")
        print(f"no noise   = {np.linalg.norm(params.θ - params_noiseless.θ)}")
        print(f"nominal    = {np.linalg.norm(params.θ - params_nom.θ)}")
        print(f"ellipsoid  = {np.linalg.norm(params.θ - params_ell.θ)}")
        print(f"polyhedron = {np.linalg.norm(params.θ - params_poly.θ)}")
        print(f"discrete   = {np.linalg.norm(params.θ - params_grid.θ)}")
        print("\nRiemannian error")
        print("-------")
        print(f"no noise   = {ip.positive_definite_distance(params.J, params_noiseless.J)}")
        print(f"nominal    = {ip.positive_definite_distance(params.J, params_nom.J)}")
        print(f"ellipsoid  = {ip.positive_definite_distance(params.J, params_ell.J)}")
        print(f"polyhedron = {ip.positive_definite_distance(params.J, params_poly.J)}")
        print(f"discrete   = {ip.positive_definite_distance(params.J, params_grid.J)}")
        print("\nvalidation error")
        print("----------------")
        print(f"no noise   = {validation_rmse(Ys_test, ws_test, params_noiseless.θ)}")
        print(f"nominal    = {validation_rmse(Ys_test, ws_test, params_nom.θ)}")
        print(f"ellipsoid  = {validation_rmse(Ys_test, ws_test, params_ell.θ)}")
        print(f"polyhedron = {validation_rmse(Ys_test, ws_test, params_poly.θ)}")
        print(f"discrete   = {validation_rmse(Ys_test, ws_test, params_grid.θ)}")

        # if i > 0:
        #     IPython.embed()
        #     raise ValueError()


main()
