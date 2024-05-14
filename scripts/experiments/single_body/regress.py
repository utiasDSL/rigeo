#!/usr/bin/env python3
"""Regress the inertial parameters from trajectory data using differently
constrained least squares problems and noise corruption."""
import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import cvxpy as cp
import tqdm

import rigeo as rg

import IPython


# fraction of data to be used for training (vs. testing)
TRAIN_TEST_SPLIT = 0.5

TRAIN_WITH_NOISY_MOTION = False

SHUFFLE = True
REGULARIZATION_COEFF = 0
PIM_EPS = 1e-4
SOLVER = cp.MOSEK


def dict_of_lists(list_keys, counter_keys):
    d1 = {key: [] for key in list_keys}
    d2 = {key: 0 for key in counter_keys}
    return {**d1, **d2}


def result_dict():
    list_keys = [
        "riemannian_errors",
        "validation_errors",
        "objective_values",
        "num_iters",
        "solve_times",
        "params",
    ]
    counter_keys = ["num_feasible"]
    return {
        "no_noise": dict_of_lists(list_keys, counter_keys),
        "nominal": dict_of_lists(list_keys, counter_keys),
        "ellipsoid": dict_of_lists(list_keys, counter_keys),
        "ell_com": dict_of_lists(list_keys, counter_keys),
        "polyhedron": dict_of_lists(list_keys, counter_keys),
    }


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="File to load the data from.")
    parser.add_argument("outfile", help="File to save the results to.")
    parser.add_argument(
        "--planar", help="Only train with planar data.", action="store_true"
    )
    parser.add_argument(
        "--bounding-box",
        help="Use bounding box rather than convex hull.",
        action="store_true",
    )
    # parser.add_argument(
    #     "--save", help="Save the results to a pickle.", action="store_true"
    # )
    args = parser.parse_args()

    with open(args.infile, "rb") as f:
        data = pickle.load(f)

    results = result_dict()
    results["use_planar"] = args.planar
    results["use_bounding_box"] = args.bounding_box
    results["num_obj"] = data["num_obj"]
    results["vel_noise_width"] = data["vel_noise_width"]
    results["vel_noise_bias"] = data["vel_noise_bias"]
    results["wrench_noise_cov"] = data["wrench_noise_cov"]
    results["wrench_noise_bias"] = data["wrench_noise_bias"]

    Σ = data["wrench_noise_cov"]
    W = np.linalg.inv(Σ)

    for i in tqdm.tqdm(range(data["num_obj"])):
        params = data["params"][i]  # true parameters
        vertices = data["vertices"][i]

        if args.bounding_box:
            box = data["bounding_box"]
            # box = rg.ConvexPolyhedron.from_vertices(vertices).aabb()
            body = rg.RigidBody(box, params)
            body_mbe = rg.RigidBody(box.mbe(), params)
        else:
            poly = rg.ConvexPolyhedron.from_vertices(vertices)
            body = rg.RigidBody(poly, params)
            body_mbe = rg.RigidBody(poly.mbe(), params)

        # ensure bounding ellipsoid is indeed bounding
        assert body_mbe.shapes[0].contains_polyhedron(body.shapes[0])

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
        if args.planar:
            Vs_noisy = np.array(data["obj_data_planar"][i]["Vs_noisy"])[idx, :]
            As_noisy = np.array(data["obj_data_planar"][i]["As_noisy"])[idx, :]
            ws_noisy = np.array(data["obj_data_planar"][i]["ws_noisy"])[idx, :]
        else:
            Vs_noisy = np.array(data["obj_data_full"][i]["Vs_noisy"])[idx, :]
            As_noisy = np.array(data["obj_data_full"][i]["As_noisy"])[idx, :]
            ws_noisy = np.array(data["obj_data_full"][i]["ws_noisy"])[idx, :]

        if TRAIN_WITH_NOISY_MOTION:
            Vs_train = Vs_noisy[:n_train]
            As_train = As_noisy[:n_train]
        else:
            Vs_train = Vs[:n_train]
            As_train = As[:n_train]
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
        res_nom = rg.IdentificationProblem(
            As=Ys_train,
            bs=ws_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            Σ=Σ,
            solver=SOLVER,
            warm_start=False,
        ).solve([body], must_realize=False)

        res_poly = rg.IdentificationProblem(
            As=Ys_train,
            bs=ws_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            Σ=Σ,
            solver=SOLVER,
            warm_start=False,
        ).solve([body], must_realize=True)

        res_ell = rg.IdentificationProblem(
            As=Ys_train,
            bs=ws_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            Σ=Σ,
            solver=SOLVER,
            warm_start=False,
        ).solve([body_mbe], must_realize=True)

        res_ell_com = rg.IdentificationProblem(
            As=Ys_train,
            bs=ws_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            Σ=Σ,
            solver=SOLVER,
            warm_start=False,
        ).solve([body_mbe], must_realize=True, com_bounding_shapes=[body.shapes[0]])

        params_nom = res_nom.params[0]
        params_ell = res_ell.params[0]
        params_ell_com = res_ell_com.params[0]
        params_poly = res_poly.params[0]

        # parameter values
        results["no_noise"]["params"].append(params_noiseless)
        results["nominal"]["params"].append(params_nom)
        results["ellipsoid"]["params"].append(params_ell)
        results["ell_com"]["params"].append(params_ell_com)
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
        results["ell_com"]["riemannian_errors"].append(
            rg.positive_definite_distance(params.J, params_ell_com.J)
        )
        results["polyhedron"]["riemannian_errors"].append(
            rg.positive_definite_distance(params.J, params_poly.J)
        )

        # validation errors
        results["no_noise"]["validation_errors"].append(
            rg.validation_rmse(Ys_test, ws_test, params_noiseless.vec)
        )
        results["nominal"]["validation_errors"].append(
            rg.validation_rmse(Ys_test, ws_test, params_nom.vec, W=None)
        )
        results["ellipsoid"]["validation_errors"].append(
            rg.validation_rmse(Ys_test, ws_test, params_ell.vec, W=None)
        )
        results["ell_com"]["validation_errors"].append(
            rg.validation_rmse(Ys_test, ws_test, params_ell_com.vec, W=None)
        )
        results["polyhedron"]["validation_errors"].append(
            rg.validation_rmse(Ys_test, ws_test, params_poly.vec, W=None)
        )

        # if (
        #     results["polyhedron"]["validation_errors"][-1]
        #     / results["nominal"]["validation_errors"][-1]
        #     > 1.2
        # ):
        #     IPython.embed()

        # objective values
        results["no_noise"]["objective_values"].append(res_noiseless.objective)
        results["nominal"]["objective_values"].append(res_nom.objective)
        results["ellipsoid"]["objective_values"].append(res_ell.objective)
        results["ell_com"]["objective_values"].append(res_ell_com.objective)
        results["polyhedron"]["objective_values"].append(res_poly.objective)

        # number of iterations
        results["no_noise"]["num_iters"].append(res_noiseless.iters)
        results["nominal"]["num_iters"].append(res_nom.iters)
        results["ellipsoid"]["num_iters"].append(res_ell.iters)
        results["ell_com"]["num_iters"].append(res_ell_com.iters)
        results["polyhedron"]["num_iters"].append(res_poly.iters)

        # solve times
        results["no_noise"]["solve_times"].append(res_noiseless.solve_time)
        results["nominal"]["solve_times"].append(res_nom.solve_time)
        results["ellipsoid"]["solve_times"].append(res_ell.solve_time)
        results["ell_com"]["solve_times"].append(res_ell_com.solve_time)
        results["polyhedron"]["solve_times"].append(res_poly.solve_time)

        poly = body.shapes[0]

        try:
            if poly.can_realize(params_nom):
                results["nominal"]["num_feasible"] += 1
        except cp.error.SolverError:
            # take failure to solve as infeasible
            pass

        try:
            if poly.can_realize(params_ell):
                results["ellipsoid"]["num_feasible"] += 1
        except cp.error.SolverError:
            pass

        try:
            if poly.can_realize(params_ell_com):
                results["ell_com"]["num_feasible"] += 1
        except cp.error.SolverError:
            pass

        assert poly.can_realize(params_poly)
        results["polyhedron"]["num_feasible"] += 1

    # save the results
    # if args.save:
    #     traj = "planar" if args.planar else "full"
    #     box = "_box" if args.bounding_box else ""
    #     w = data["vel_noise_width"]
    #     b = data["vel_noise_bias"]
    #     outfile = Path(args.infile).parent / f"regress_w{w}_b{b}_{traj}{box}.pkl"
    #
    #     with open(outfile, "wb") as f:
    #         pickle.dump(results, f)
    #     print(f"Saved results to {outfile}")

    with open(args.outfile, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {args.outfile}")


main()
