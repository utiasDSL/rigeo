#!/usr/bin/env python3
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

import rigeo as rg

import IPython


# these values are taken from the Robotiq FT-300 datasheet
# WRENCH_STDEV = np.array([1.2, 1.2, 0.5, 0.02, 0.02, 0.03])

# DISC_GRID_SIZE = 5

# fraction of data to be used for training (vs. testing)
TRAIN_TEST_SPLIT = 0.2

# train only with data in the x-y plane
TRAIN_WITH_PLANAR_ONLY = False

# use the overall bounding box rather than tight convex hull of the body's
# shape
# TODO this makes things worse
USE_BOUNDING_BOX = False

REGULARIZATION_COEFF = 0
PIM_EPS = 1e-4
SOLVER = cp.MOSEK


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
        # print(f"discrete   = {self.discrete[s]}")

    def print_average(self):
        print(self.name)
        print("".ljust(len(self.name), "-"))
        print(f"no noise   = {np.median(self.no_noise)}")
        print(f"nominal    = {np.median(self.nominal)}")
        print(f"ellipsoid  = {np.median(self.ellipsoid)}")
        print(f"polyhedron = {np.median(self.polyhedron)}")
        # print(f"discrete   = {np.median(self.discrete)}")

        poly = np.array(self.polyhedron)
        nom = np.array(self.nominal)
        ell = np.array(self.ellipsoid)
        n = len(self.polyhedron)
        n_poly_better_than_nom = np.sum(poly <= nom)
        n_poly_better_than_ell = np.sum(poly <= ell)
        print(f"poly better than nom: {n_poly_better_than_nom}/{n}")
        print(f"poly better than ell: {n_poly_better_than_ell}/{n}")

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

        # NOTE I am using the true params here
        if USE_BOUNDING_BOX:
            box = data["bounding_box"]
            body = rg.RigidBody(box, params)
            body_mbe = rg.RigidBody(box.mbe(), params)
        else:
            poly = rg.ConvexPolyhedron.from_vertices(vertices)
            body = rg.RigidBody(poly, params)
            body_mbe = rg.RigidBody(poly.mbe(), params)

        # Ys = np.array(data["obj_data"][i]["Ys"])
        # Ys_noisy = np.array(data["obj_data"][i]["Ys_noisy"])
        Vs = np.array(data["obj_data_full"][i]["Vs"])
        As = np.array(data["obj_data_full"][i]["As"])
        ws = np.array(data["obj_data_full"][i]["ws"])

        n = Vs.shape[0]
        n_train = int(TRAIN_TEST_SPLIT * n)
        # n_train = 1
        # cov = np.eye(6)  # TODO?

        # regression/training data
        if TRAIN_WITH_PLANAR_ONLY:
            Vs_noisy = np.array(data["obj_data_planar"][i]["Vs_noisy"])
            As_noisy = np.array(data["obj_data_planar"][i]["As_noisy"])
            ws_noisy = np.array(data["obj_data_planar"][i]["ws_noisy"])
        else:
            Vs_noisy = np.array(data["obj_data_full"][i]["Vs_noisy"])
            As_noisy = np.array(data["obj_data_full"][i]["As_noisy"])
            ws_noisy = np.array(data["obj_data_full"][i]["ws_noisy"])

        Vs_train = Vs_noisy[:n_train]
        As_train = As_noisy[:n_train]
        ws_train = ws_noisy[:n_train]

        Ys_train = np.array(
            [rg.RigidBody.regressor(V, A) for V, A in zip(Vs_train, As_train)]
        )

        # test/validation data
        # TODO not sure if we should use the noisy or noiseless data for
        # testing -- noiseless is basically ground-truth
        Vs_test = Vs[n_train:]
        As_test = As[n_train:]
        Ys_test = np.array(
            [rg.RigidBody.regressor(V, A) for V, A in zip(Vs_test, As_test)]
        )
        # ws_test = ws_noisy[n_train:-1]
        ws_test = ws[n_train:]

        # solve the problem with no noise (just to make sure things are working)
        Ys_train_noiseless = np.array(
            [rg.RigidBody.regressor(V, A) for V, A in zip(Vs, As)]
        )[:n_train]
        ws_train_noiseless = ws[:n_train]

        # TODO support covariance again
        prob_noiseless = rg.IdentificationProblem(
            As=Ys_train_noiseless,
            bs=ws_train_noiseless,
            solver=SOLVER,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
        )
        params_noiseless = prob_noiseless.solve([body], must_realize=False)[0]

        # solve noisy problem with varying constraints
        prob = rg.IdentificationProblem(
            As=Ys_train,
            bs=ws_train,
            solver=SOLVER,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
        )
        params_nom = prob.solve([body], must_realize=False)[0]
        params_poly = prob.solve([body], must_realize=True)[0]
        params_ell = prob.solve([body_mbe], must_realize=True)[0]
        # params_both = prob.solve_both(vertices, ellipsoid)

        # regularize with equal point masses
        # params_grid = DiscretizedIPIDProblem(
        #     Ys_train, ws_train, cov, reg_masses, reg_coeff=REGULARIZATION_COEFF
        # ).solve(grid)

        θ_errors.no_noise.append(np.linalg.norm(params.vec - params_noiseless.vec))
        θ_errors.nominal.append(np.linalg.norm(params.vec - params_nom.vec))
        θ_errors.ellipsoid.append(np.linalg.norm(params.vec - params_ell.vec))
        θ_errors.polyhedron.append(np.linalg.norm(params.vec - params_poly.vec))
        # θ_errors.discrete.append(np.linalg.norm(params.θ - params_grid.θ))

        riemannian_errors.no_noise.append(
            rg.positive_definite_distance(params.J, params_noiseless.J)
        )
        riemannian_errors.nominal.append(
            rg.positive_definite_distance(params.J, params_nom.J)
        )
        riemannian_errors.ellipsoid.append(
            rg.positive_definite_distance(params.J, params_ell.J)
        )
        riemannian_errors.polyhedron.append(
            rg.positive_definite_distance(params.J, params_poly.J)
        )
        # riemannian_errors.discrete.append(
        #     rg.positive_definite_distance(params.J, params_grid.J)
        # )

        validation_errors.no_noise.append(
            rg.validation_rmse(Ys_test, ws_test, params_noiseless.vec)
        )
        validation_errors.nominal.append(
            rg.validation_rmse(Ys_test, ws_test, params_nom.vec)
        )
        validation_errors.ellipsoid.append(
            rg.validation_rmse(Ys_test, ws_test, params_ell.vec)
        )
        validation_errors.polyhedron.append(
            rg.validation_rmse(Ys_test, ws_test, params_poly.vec)
        )
        # validation_errors.discrete.append(
        #     rg.validation_rmse(Ys_test, ws_test, params_grid.θ)
        # )

        print(f"\nProblem {i + 1}")
        print("==========")
        print(f"nv = {vertices.shape[0]}")
        # print(f"ng = {grid.shape[0]}")
        # print()
        # θ_errors.print(index=i)
        print()
        riemannian_errors.print(index=i)
        print()
        validation_errors.print(index=i)

        # if i == 19:
        #     IPython.embed()

    print(f"\nMedians")
    print("========")
    # print()
    # θ_errors.print_average()
    print()
    riemannian_errors.print_average()
    print()
    validation_errors.print_average()


main()
