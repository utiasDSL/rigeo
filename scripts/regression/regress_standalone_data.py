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
# WRENCH_STDEV = np.array([1.2, 1.2, 0.5, 0.02, 0.02, 0.03])

DISC_GRID_SIZE = 2
TRAIN_TEST_SPLIT = 0.5
TRAIN_WITH_NOISY_Y = True
TRAIN_WITH_NOISY_W = True
USE_BOUNDING_BOX = True

# TODO: test discrete with just vertices includes (and maybe the centroid)


def J_vec_constraint(J, θ, eps=1e-4):
    """Constraint to enforce consistency between J and θ representations."""
    H = J[:3, :3]
    I = cp.trace(H) * np.eye(3) - H
    return [
        J >> eps * np.eye(4),
        J[3, 3] == θ[0],
        J[:3, 3] == θ[1:4],
        I[0, :3] == θ[4:7],
        I[1, 1:3] == θ[7:9],
        I[2, 2] == θ[9],
    ]


def schur(X, x, m):
    y = cp.reshape(x, (x.shape[0], 1))
    z = cp.reshape(m, (1, 1))
    return cp.bmat([[X, y], [y.T, z]])


class DiscretizedIPIDProblem:
    def __init__(self, Ys, ws, cov):
        self.A = np.vstack(Ys)
        self.b = np.concatenate(ws)
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

    def __init__(self, Ys, ws, cov):
        self.A = np.vstack(Ys)
        self.b = np.concatenate(ws)
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
        assert problem.status == "optimal"
        # if name is not None:
        #     # print(f"{name} solve time = {problem.solver_stats.solve_time}")
        #     print(f"{name} value = {problem.value}")
        return ip.RigidBody.from_vector(self.θ.value)
        # return ip.RigidBody.from_pseudo_inertia_matrix(self.Jopt.value)

    def solve_nominal(self):
        return self._solve()

    def solve_ellipsoid(self, ellipsoid):
        extra_constraints = [cp.trace(ellipsoid.Q @ self.Jopt) >= 0]
        return self._solve(extra_constraints)

    def solve_polyhedron(self, vertices):
        nv = vertices.shape[0]
        mvs = cp.Variable(nv)
        Vs = np.array([np.outer(v, v) for v in vertices])
        extra_constraints = [
            mvs >= 0,
            cp.sum(mvs) == self.θ[0],
            mvs.T @ vertices == self.θ[1:4],
            self.Jopt[:3, :3] << cp.sum([m * V for m, V in zip(mvs, Vs)]),
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
    """Compute root mean square wrench error on a validation set."""
    error = Ys @ θ - ws
    square = np.sum(error**2, axis=1)
    mean = np.mean(square)
    root = np.sqrt(mean)
    return root


class ErrorSet:
    def __init__(self, name):
        self.name = name

        self.no_noise = []
        self.nominal = []
        self.ellipsoid = []
        self.polyhedron = []
        self.discrete = []

    def print(self, index=None):
        print(self.name)
        print("".ljust(len(self.name), "-"))

        if index is None:
            s = np.s_[:]
        else:
            s = np.s_[index]

        print(f"no noise   = {self.no_noise[s]}")
        print(f"nominal    = {self.nominal[s]}")
        print(f"ellipsoid  = {self.ellipsoid[s]}")
        print(f"polyhedron = {self.polyhedron[s]}")
        print(f"discrete   = {self.discrete[s]}")

    def print_average(self):
        print(self.name)
        print("".ljust(len(self.name), "-"))
        print(f"no noise   = {np.mean(self.no_noise)}")
        print(f"nominal    = {np.mean(self.nominal)}")
        print(f"ellipsoid  = {np.mean(self.ellipsoid)}")
        print(f"polyhedron = {np.mean(self.polyhedron)}")
        print(f"discrete   = {np.mean(self.discrete)}")


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="File to load the data from.")
    args = parser.parse_args()

    with open(args.infile, "rb") as f:
        data = pickle.load(f)

    θ_errors = ErrorSet("θ error")
    riemannian_errors = ErrorSet("Riemannian error")
    validation_errors = ErrorSet("Validation error")

    for i in range(data["num_obj"]):
        params = data["params"][i]
        vertices = data["vertices"][i]

        if USE_BOUNDING_BOX:
            box = ip.AxisAlignedBox.from_points_to_bound(vertices)
            vertices = box.vertices
            grid = box.grid(n=DISC_GRID_SIZE)
        else:
            grid = ip.polyhedron_grid(vertices, n=DISC_GRID_SIZE)

        ellipsoid = ip.minimum_bounding_ellipsoid(vertices)
        # IPython.embed()
        # return

        Ys = np.array(data["obj_data"][i]["Ys"])
        Ys_noisy = np.array(data["obj_data"][i]["Ys_noisy"])
        ws = np.array(data["obj_data"][i]["ws"])
        ws_noisy = np.array(data["obj_data"][i]["ws_noisy"])

        n = Ys.shape[0]
        n_train = int(TRAIN_TEST_SPLIT * n)

        # regression data
        if TRAIN_WITH_NOISY_Y:
            Ys_train = Ys_noisy[:n_train]
            Ys_test = Ys_noisy[n_train:]
        else:
            Ys_train = Ys[:n_train]
            Ys_test = Ys[n_train:-1]  # NOTE

        if TRAIN_WITH_NOISY_W:
            ws_train = ws_noisy[:n_train]
            ws_test = ws_noisy[n_train:-1]
            # cov = np.diag(WRENCH_STDEV**2)
            cov = np.eye(6)  # TODO?
        else:
            ws_train = ws[:n_train]
            ws_test = ws[n_train:-1]
            cov = np.eye(6)

        # solve the problem with no noise
        prob_noiseless = IPIDProblem(Ys[:n_train], ws[:n_train], np.eye(6))
        params_noiseless = prob_noiseless.solve_nominal()

        # solve noisy problem with varying constraints
        prob = IPIDProblem(Ys_train, ws_train, cov)
        params_nom = prob.solve_nominal()
        params_poly = prob.solve_polyhedron(vertices)
        params_ell = prob.solve_ellipsoid(ellipsoid)
        params_both = prob.solve_both(vertices, ellipsoid)

        params_grid = DiscretizedIPIDProblem(Ys_train, ws_train, cov).solve(grid)

        # TODO maybe should use noisy values?

        θ_errors.no_noise.append(np.linalg.norm(params.θ - params_noiseless.θ))
        θ_errors.nominal.append(np.linalg.norm(params.θ - params_nom.θ))
        θ_errors.ellipsoid.append(np.linalg.norm(params.θ - params_ell.θ))
        θ_errors.polyhedron.append(np.linalg.norm(params.θ - params_poly.θ))
        θ_errors.discrete.append(np.linalg.norm(params.θ - params_grid.θ))

        riemannian_errors.no_noise.append(
            ip.positive_definite_distance(params.J, params_noiseless.J)
        )
        riemannian_errors.nominal.append(
            ip.positive_definite_distance(params.J, params_nom.J)
        )
        riemannian_errors.ellipsoid.append(
            ip.positive_definite_distance(params.J, params_ell.J)
        )
        riemannian_errors.polyhedron.append(
            ip.positive_definite_distance(params.J, params_poly.J)
        )
        riemannian_errors.discrete.append(
            ip.positive_definite_distance(params.J, params_grid.J)
        )

        validation_errors.no_noise.append(
            validation_rmse(Ys_test, ws_test, params_noiseless.θ)
        )
        validation_errors.nominal.append(
            validation_rmse(Ys_test, ws_test, params_nom.θ)
        )
        validation_errors.ellipsoid.append(
            validation_rmse(Ys_test, ws_test, params_ell.θ)
        )
        validation_errors.polyhedron.append(
            validation_rmse(Ys_test, ws_test, params_poly.θ)
        )
        validation_errors.discrete.append(
            validation_rmse(Ys_test, ws_test, params_grid.θ)
        )

        print(f"\nProblem {i + 1}")
        print("==========")
        print(f"nv = {vertices.shape[0]}")
        print(f"ng = {grid.shape[0]}")
        print()
        θ_errors.print(index=i)
        print()
        riemannian_errors.print(index=i)
        print()
        validation_errors.print(index=i)

        if i == 9:
            IPython.embed()

    print(f"\nAverages")
    print("========")
    print()
    θ_errors.print_average()
    print()
    riemannian_errors.print_average()
    print()
    validation_errors.print_average()


main()
