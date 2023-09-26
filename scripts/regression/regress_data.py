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


# WRENCH_STDEV = 1.0

# these values are taken from the Robotiq FT-300 datasheet
WRENCH_STDEV = np.array([1.2, 1.2, 0.5, 0.02, 0.02, 0.03])


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


class IPIDProblem:
    """Inertial parameter identification optimization problem."""

    def __init__(self, Ys, ws, ws_noise):
        self.A = np.vstack(Ys)
        self.b = np.concatenate(ws + ws_noise)
        self.θ = cp.Variable(10)
        self.Jopt = cp.Variable((4, 4), PSD=True)

    def _solve(self, extra_constraints=None):
        objective = cp.Minimize(cp.sum_squares(self.A @ self.θ - self.b))

        constraints = J_vec_constraint(self.Jopt, self.θ)
        if extra_constraints is not None:
            constraints.extend(extra_constraints)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        return ip.RigidBody.from_vector(self.θ.value)

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
        return self._solve(extra_constraints)

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
        ellipsoid = ip.minimum_bounding_ellipsoid(vertices)

        Ys = np.array(data["obj_data"][i]["Ys"])
        ws = np.array(data["obj_data"][i]["ws"])

        # generate random noise on the force measurements
        ws_noise = np.random.normal(scale=WRENCH_STDEV, size=ws.shape)

        # solve the problem with no noise
        prob_noiseless = IPIDProblem(Ys, ws, np.zeros_like(ws_noise))
        params_noiseless = prob_noiseless.solve_nominal()

        # solve noisy problem with varying constraints
        prob = IPIDProblem(Ys, ws, ws_noise)
        params_nom = prob.solve_nominal()
        params_poly = prob.solve_polyhedron(vertices)
        params_ell = prob.solve_ellipsoid(ellipsoid)
        params_both = prob.solve_both(vertices, ellipsoid)

        print(f"\nProblem {i + 1}")
        print(f"nv = {vertices.shape[0]}")
        print(f"no noise err = {np.linalg.norm(params.θ - params_noiseless.θ)}")
        print(f"nom err  = {np.linalg.norm(params.θ - params_nom.θ)}")
        print(f"ell err  = {np.linalg.norm(params.θ - params_ell.θ)}")
        print(f"poly err = {np.linalg.norm(params.θ - params_poly.θ)}")
        print(f"both err = {np.linalg.norm(params.θ - params_both.θ)}")
        #
        # if i == 6:
        #     IPython.embed()
        #     return


main()
