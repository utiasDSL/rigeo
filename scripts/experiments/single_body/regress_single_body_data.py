#!/usr/bin/env python3
"""Regress the inertial parameters from trajectory data using differently
constrained least squares problems and noise corruption."""
import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import cvxpy as cp
import colorama
import tqdm

import rigeo as rg

import IPython


# fraction of data to be used for training (vs. testing)
TRAIN_TEST_SPLIT = 0.5

# train only with data in the x-y plane
TRAIN_WITH_PLANAR_ONLY = False

# use the overall bounding box rather than tight convex hull of the body's
# shape -- this is equivalent to just having a box shape with the given params
USE_BOUNDING_BOX = False

SHUFFLE = True

REGULARIZATION_COEFF = 0
PIM_EPS = 1e-4
SOLVER = cp.MOSEK


def green(s):
    """Color a string green."""
    return colorama.Fore.GREEN + s + colorama.Fore.RESET


def yellow(s):
    """Color a string yellow."""
    return colorama.Fore.YELLOW + s + colorama.Fore.RESET


class ErrorSet:
    def __init__(self, name):
        self.name = name

        self.no_noise = []
        self.nominal = []
        self.ellipsoid = []
        self.polyhedron = []

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

    def print_average(self):
        print(self.name)
        print("".ljust(len(self.name), "-"))
        print(f"no noise   = {np.median(self.no_noise)}")
        print(f"nominal    = {np.median(self.nominal)}")
        print(f"ellipsoid  = {np.median(self.ellipsoid)}")
        print(f"polyhedron = {np.median(self.polyhedron)}")

        poly = np.array(self.polyhedron)
        nom = np.array(self.nominal)
        ell = np.array(self.ellipsoid)
        n = len(self.polyhedron)
        n_poly_lower_than_nom = np.sum(poly <= nom)
        n_poly_lower_than_ell = np.sum(poly <= ell)
        print(f"poly lower than nom: {n_poly_lower_than_nom}/{n}")
        print(f"poly lower than ell: {n_poly_lower_than_ell}/{n}")


def dict_of_lists(keys):
    return {key: [] for key in keys}


def result_dict():
    stats = [
        "riemannian_errors",
        "validation_errors",
        "objective_values",
        "num_iters",
        "solve_times",
        "params",
    ]
    return {
        "no_noise": dict_of_lists(stats),
        "nominal": dict_of_lists(stats),
        "ellipsoid": dict_of_lists(stats),
        "polyhedron": dict_of_lists(stats),
    }


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="File to load the data from.")
    parser.add_argument("outfile", help="File to save the results to.")
    args = parser.parse_args()

    with open(args.infile, "rb") as f:
        data = pickle.load(f)

    # riemannian_errors = ErrorSet("Riemannian error")
    # validation_errors = ErrorSet("Validation error")
    # objective_values = ErrorSet("Objective value")
    # num_iterations = ErrorSet("Num iterations")
    # solve_times = ErrorSet("Solve time")

    results = result_dict()
    results["data"] = data  # save the data as well

    for i in tqdm.tqdm(range(data["num_obj"])):
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

        idx = np.arange(data["obj_data_full"][i]["Vs"].shape[0])
        if SHUFFLE:
            np.random.shuffle(idx)

        # ground truth
        Vs = np.array(data["obj_data_full"][i]["Vs"])[idx, :]
        As = np.array(data["obj_data_full"][i]["As"])[idx, :]
        ws = np.array(data["obj_data_full"][i]["ws"])[idx, :]

        n = Vs.shape[0]
        n_train = int(TRAIN_TEST_SPLIT * n)

        # regression/training data
        if TRAIN_WITH_PLANAR_ONLY:
            Vs_noisy = np.array(data["obj_data_planar"][i]["Vs_noisy"])[idx, :]
            As_noisy = np.array(data["obj_data_planar"][i]["As_noisy"])[idx, :]
            ws_noisy = np.array(data["obj_data_planar"][i]["ws_noisy"])[idx, :]
        else:
            Vs_noisy = np.array(data["obj_data_full"][i]["Vs_noisy"])[idx, :]
            As_noisy = np.array(data["obj_data_full"][i]["As_noisy"])[idx, :]
            ws_noisy = np.array(data["obj_data_full"][i]["ws_noisy"])[idx, :]

        Vs_train = Vs_noisy[:n_train]
        As_train = As_noisy[:n_train]
        ws_train = ws_noisy[:n_train]

        Ys_train = np.array(
            [rg.RigidBody.regressor(V, A) for V, A in zip(Vs_train, As_train)]
        )

        # test/validation data: use noiseless ground-truth
        Vs_test = Vs[n_train:]
        As_test = As[n_train:]
        Ys_test = np.array(
            [rg.RigidBody.regressor(V, A) for V, A in zip(Vs_test, As_test)]
        )
        ws_test = ws[n_train:]

        # solve the problem with no noise (just to make sure things are working)
        Ys_train_noiseless = np.array(
            [rg.RigidBody.regressor(V, A) for V, A in zip(Vs, As)]
        )[:n_train]
        ws_train_noiseless = ws[:n_train]

        prob_noiseless = rg.IdentificationProblem(
            As=Ys_train_noiseless,
            bs=ws_train_noiseless,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            solver=SOLVER,
            warm_start=False,
        )
        res_noiseless = prob_noiseless.solve([body], must_realize=False)
        params_noiseless = res_noiseless.params[0]

        # solve noisy problem with varying constraints
        prob = rg.IdentificationProblem(
            As=Ys_train,
            bs=ws_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            solver=SOLVER,
            warm_start=False,
        )

        res_nom = prob.solve([body], must_realize=False)
        res_poly = prob.solve([body], must_realize=True)
        res_ell = prob.solve([body_mbe], must_realize=True)

        params_nom = res_nom.params[0]
        params_poly = res_poly.params[0]
        params_ell = res_ell.params[0]

        # parameter values
        results["no_noise"]["params"].append(params_noiseless)
        results["nominal"]["params"].append(params_nom)
        results["ellipsoid"]["params"].append(params_ell)
        results["polyhedron"]["params"].append(params_poly)

        # Riemannian errors
        results["no_noise"]["riemannian_errors"].append(
            rg.positive_definite_distance(params.J, params_noiseless.J)
        )
        results["nominal"]["riemannian_errors"].append(
            rg.positive_definite_distance(params.J, params_nom.J)
        )
        results["ellipsoid"]["riemannian_errors"].append(
            rg.positive_definite_distance(params.J, params_ell.J)
        )
        results["polyhedron"]["riemannian_errors"].append(
            rg.positive_definite_distance(params.J, params_poly.J)
        )

        # validation errors
        results["no_noise"]["validation_errors"].append(
            rg.validation_rmse(Ys_test, ws_test, params_noiseless.vec)
        )
        results["nominal"]["validation_errors"].append(
            rg.validation_rmse(Ys_test, ws_test, params_nom.vec)
        )
        results["ellipsoid"]["validation_errors"].append(
            rg.validation_rmse(Ys_test, ws_test, params_ell.vec)
        )
        results["polyhedron"]["validation_errors"].append(
            rg.validation_rmse(Ys_test, ws_test, params_poly.vec)
        )

        # objective values
        results["no_noise"]["objective_values"].append(res_noiseless.objective)
        results["nominal"]["objective_values"].append(res_nom.objective)
        results["ellipsoid"]["objective_values"].append(res_ell.objective)
        results["polyhedron"]["objective_values"].append(res_poly.objective)

        # number of iterations
        results["no_noise"]["num_iters"].append(res_noiseless.iters)
        results["nominal"]["num_iters"].append(res_nom.iters)
        results["ellipsoid"]["num_iters"].append(res_ell.iters)
        results["polyhedron"]["num_iters"].append(res_poly.iters)

        # solve times
        results["no_noise"]["solve_times"].append(res_noiseless.solve_time)
        results["nominal"]["solve_times"].append(res_nom.solve_time)
        results["ellipsoid"]["solve_times"].append(res_ell.solve_time)
        results["polyhedron"]["solve_times"].append(res_poly.solve_time)

        # riemannian_errors.no_noise.append(
        #     rg.positive_definite_distance(params.J, params_noiseless.J)
        # )
        # riemannian_errors.nominal.append(
        #     rg.positive_definite_distance(params.J, params_nom.J)
        # )
        # riemannian_errors.ellipsoid.append(
        #     rg.positive_definite_distance(params.J, params_ell.J)
        # )
        # riemannian_errors.polyhedron.append(
        #     rg.positive_definite_distance(params.J, params_poly.J)
        # )

        # validation_errors.no_noise.append(
        #     rg.validation_rmse(Ys_test, ws_test, params_noiseless.vec)
        # )
        # validation_errors.nominal.append(
        #     rg.validation_rmse(Ys_test, ws_test, params_nom.vec)
        # )
        # validation_errors.ellipsoid.append(
        #     rg.validation_rmse(Ys_test, ws_test, params_ell.vec)
        # )
        # validation_errors.polyhedron.append(
        #     rg.validation_rmse(Ys_test, ws_test, params_poly.vec)
        # )

        # objective_values.no_noise.append(res_noiseless.objective)
        # objective_values.nominal.append(res_nom.objective)
        # objective_values.polyhedron.append(res_poly.objective)
        # objective_values.ellipsoid.append(res_ell.objective)
        #
        # num_iterations.no_noise.append(res_noiseless.iters)
        # num_iterations.nominal.append(res_nom.iters)
        # num_iterations.polyhedron.append(res_poly.iters)
        # num_iterations.ellipsoid.append(res_ell.iters)
        #
        # solve_times.no_noise.append(res_noiseless.solve_time)
        # solve_times.nominal.append(res_nom.solve_time)
        # solve_times.polyhedron.append(res_poly.solve_time)
        # solve_times.ellipsoid.append(res_ell.solve_time)

        # s = yellow(f"Problem {i + 1}" + "\n==========")
        # print("\n" + s)
        # print(f"nv = {vertices.shape[0]}")
        # print()
        # riemannian_errors.print(index=i)
        # print()
        # validation_errors.print(index=i)
        # print()
        # objective_values.print(index=i)
        # print()
        # num_iterations.print(index=i)
        # print()
        # solve_times.print(index=i)

    # s = green("Medians\n=======")
    # print("\n" + s + "\n")
    # riemannian_errors.print_average()
    # print()
    # validation_errors.print_average()
    # print()
    # objective_values.print_average()
    # print()
    # num_iterations.print_average()
    # print()
    # solve_times.print_average()

    # save the results
    with open(args.outfile, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {args.outfile}")


main()
