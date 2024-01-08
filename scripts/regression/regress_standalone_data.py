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
from scipy.linalg import block_diag

import inertial_params as ip

import IPython


# these values are taken from the Robotiq FT-300 datasheet
# WRENCH_STDEV = np.array([1.2, 1.2, 0.5, 0.02, 0.02, 0.03])

DISC_GRID_SIZE = 2

# fraction of data to be used for training (vs. testing)
TRAIN_TEST_SPLIT = 0.2

# train only with data in the x-y plane
TRAIN_WITH_PLANAR_ONLY = False

# use the overall bounding box rather than tight convex hull of the body's
# shape
USE_BOUNDING_BOX = True


def J_vec_constraint(J, θ, eps=1e-2):
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
    """Inertial parameter identification optimization using a discrete set of point masses."""

    def __init__(self, Ys, ws, cov, reg_masses, reg_coeff=1e-3):
        self.n = ws.shape[0]  # number of measurements
        self.A = np.vstack(Ys)
        self.b = np.concatenate(ws)
        self.θ = cp.Variable(10)
        self.Jopt = cp.Variable((4, 4), PSD=True)

        self.reg_masses = reg_masses
        self.reg_coeff = reg_coeff

        cov_inv = np.linalg.inv(cov)
        self.W = np.kron(np.eye(Ys.shape[0]), cov_inv)

    def solve(self, points):
        masses = cp.Variable(points.shape[0])
        regularizer = self.reg_coeff * cp.quad_form(
            masses - self.reg_masses, np.eye(points.shape[0])
        )
        objective = cp.Minimize(
            0.5 / self.n * cp.quad_form(self.A @ self.θ - self.b, self.W) + regularizer
        )
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
    """Inertial parameter identification optimization problem.

    Parameters
    ----------
    Ys : ndarray, shape (n, 10, 6)
        Regressor matrices, one for each of the ``n`` measurements.
    ws : ndarray, shape (n, 6)
        Measured wrenches.
    cov : ndarray, shape (6, 6)
        Covariance matrix for the measurement noise.
    reg_params : RigidBody
        Nominal inertial parameters to use as a regularizer.
    reg_coeff : float
        Coefficient for the regularization term.
    """

    def __init__(self, Ys, ws, cov, reg_params, reg_coeff=1e-3):
        self.n = ws.shape[0]  # number of measurements
        self.A = np.vstack(Ys)
        self.b = np.concatenate(ws)
        self.θ = cp.Variable(10)
        self.Jopt = cp.Variable((4, 4), PSD=True)

        self.J0_inv = np.linalg.inv(reg_params.J)
        self.reg_coeff = reg_coeff

        cov_inv = np.linalg.inv(cov)
        self.W = np.kron(np.eye(Ys.shape[0]), cov_inv)

    def _solve(self, extra_constraints=None, name=None):
        # regularizer is the entropic distance proposed by (Lee et al., 2020)
        regularizer = -cp.log_det(self.Jopt) + cp.trace(self.J0_inv @ self.Jopt)
        objective = cp.Minimize(
            0.5 / self.n * cp.quad_form(self.A @ self.θ - self.b, self.W)
            + self.reg_coeff * regularizer
        )

        constraints = J_vec_constraint(self.Jopt, self.θ)
        if extra_constraints is not None:
            constraints.extend(extra_constraints)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        assert problem.status == "optimal"
        if name is not None:
            print(f"{name} solve time = {problem.solver_stats.solve_time}")
        #     print(f"{name} value = {problem.value}")
        return ip.RigidBody.from_vector(self.θ.value)
        # return ip.RigidBody.from_pseudo_inertia_matrix(self.Jopt.value)

    def solve_nominal(self):
        return self._solve()

    def solve_ellipsoid(self, ellipsoid):
        extra_constraints = [cp.trace(ellipsoid.Q @ self.Jopt) >= 0]
        return self._solve(extra_constraints, name="ellipsoid")

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

    # def save(self, filename):
    #     with open(filename, "w") as f:
    #         pass


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
            # box = ip.AxisAlignedBox.from_points_to_bound(vertices)
            box = data["bounding_box"]
            vertices = box.vertices
            grid = box.grid(n=DISC_GRID_SIZE)
        else:
            grid = ip.polyhedron_grid(vertices, n=DISC_GRID_SIZE)

        ellipsoid = ip.minimum_bounding_ellipsoid(vertices)

        # Ys = np.array(data["obj_data"][i]["Ys"])
        # Ys_noisy = np.array(data["obj_data"][i]["Ys_noisy"])
        Vs = np.array(data["obj_data"][i]["Vs"])
        Vs_noisy = np.array(data["obj_data"][i]["Vs_noisy"])
        As = np.array(data["obj_data"][i]["As"])
        As_noisy = np.array(data["obj_data"][i]["As_noisy"])
        ws = np.array(data["obj_data"][i]["ws"])
        ws_noisy = np.array(data["obj_data"][i]["ws_noisy"])

        n = Vs.shape[0]
        n_train = int(TRAIN_TEST_SPLIT * n)

        # regression/training data
        Vs_train = Vs_noisy[:n_train]
        As_train = As_noisy[:n_train]
        ws_train = ws_noisy[:n_train]
        cov = np.eye(6)  # TODO?

        if TRAIN_WITH_PLANAR_ONLY:
            C_wbs = np.array(data["obj_data"][i]["C_wbs"])
            for j in range(n_train):
                C_wb = C_wbs[j]
                X_wb = block_diag(C_wb, C_wb)
                V_w = X_wb @ Vs_train[j]
                A_w = X_wb @ As_train[j]
                w_w = X_wb @ ws_train[j]

                V_w[2:5] = 0
                A_w[2:5] = 0
                w_w[2:5] = 0

                Vs_train[j] = X_wb.T @ V_w
                As_train[j] = X_wb.T @ A_w
                ws_train[j] = X_wb.T @ w_w

            # Vs_train[:, 2:5] = 0
            # As_train[:, 2:5] = 0
            # ws_train[:, 2:5] = 0

        Ys_train = np.array(
            [ip.body_regressor(V, A) for V, A in zip(Vs_train, As_train)]
        )

        # test/validation data
        Vs_test = Vs[n_train:]
        As_test = As[n_train:]
        Ys_test = np.array([ip.body_regressor(V, A) for V, A in zip(Vs_test, As_test)])
        # ws_test = ws_noisy[n_train:]
        ws_test = ws[n_train:]

        # solve the problem with no noise (just to make sure things are working)
        Ys_train_noiseless = np.array(
            [ip.body_regressor(V, A) for V, A in zip(Vs, As)]
        )[:n_train]
        ws_train_noiseless = ws[:n_train]
        prob_noiseless = IPIDProblem(
            Ys_train_noiseless, ws_train_noiseless, np.eye(6), reg_params=params
        )
        try:
            params_noiseless = prob_noiseless.solve_nominal()
        except:
            print("failed to solve noiseless problem")
            params_noiseless = ip.RigidBody.from_pseudo_inertia_matrix(params.J)

        # solve noisy problem with varying constraints
        prob = IPIDProblem(Ys_train, ws_train, cov, reg_params=params)
        params_nom = prob.solve_nominal()
        params_poly = prob.solve_polyhedron(vertices)
        params_ell = prob.solve_ellipsoid(ellipsoid)
        params_both = prob.solve_both(vertices, ellipsoid)

        # regularize with equal point masses
        # TODO this probably doesn't really make sense
        reg_masses = params.mass * np.ones(grid.shape[0]) / grid.shape[0]
        params_grid = DiscretizedIPIDProblem(Ys_train, ws_train, cov, reg_masses).solve(
            grid
        )

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
        # print()
        # θ_errors.print(index=i)
        print()
        riemannian_errors.print(index=i)
        print()
        validation_errors.print(index=i)

        # if i == 19:
        #     IPython.embed()

    print(f"\nAverages")
    print("========")
    # print()
    # θ_errors.print_average()
    print()
    riemannian_errors.print_average()
    print()
    validation_errors.print_average()


main()
