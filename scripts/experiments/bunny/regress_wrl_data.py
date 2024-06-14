#!/usr/bin/env python3
"""Regress the inertial parameters from trajectory data using differently
constrained least squares problems and noise corruption."""
import argparse
import pickle

import numpy as np
import cvxpy as cp
import tqdm

import rigeo as rg


TRAIN_TEST_SPLIT = 0.5
TRAIN_WITH_NOISY_MOTION = False
SHUFFLE = True
REGULARIZATION_COEFF = 0
PIM_EPS = 1e-4
SOLVER = cp.MOSEK


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("wrlfile", help="WRL/VRML file to load.")
    parser.add_argument("pklfile", help="Pickle file to load the data from.")
    args = parser.parse_args()

    # load the data
    with open(args.pklfile, "rb") as f:
        data = pickle.load(f)

    # load the scene
    scene = rg.WRL.from_file_path(args.wrlfile, diaglen=data["bb_diag_len"])

    Σ = data["wrench_noise_cov"]

    regression_results = {
        "num_iters": [],
        "solve_times": [],
        "params": [],
    }

    verification_results = {
        "num_iters": [],
        "solve_times": [],
    }

    for i in tqdm.tqdm(range(data["num_params"])):
        params = data["params"][i]
        body = scene.body(params)

        idx = np.arange(data["obj_data_full"][i]["Vs"].shape[0])
        if SHUFFLE:
            np.random.shuffle(idx)

        # ground truth
        Vs = np.array(data["obj_data_full"][i]["Vs"])[idx, :]
        As = np.array(data["obj_data_full"][i]["As"])[idx, :]
        ws = np.array(data["obj_data_full"][i]["ws"])[idx, :]

        # noisy data
        Vs_noisy = np.array(data["obj_data_full"][i]["Vs_noisy"])[idx, :]
        As_noisy = np.array(data["obj_data_full"][i]["As_noisy"])[idx, :]
        ws_noisy = np.array(data["obj_data_full"][i]["ws_noisy"])[idx, :]

        n = Vs.shape[0]
        n_train = int(TRAIN_TEST_SPLIT * n)

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

        # test/validation data
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

        id_res = rg.IdentificationProblem(
            As=Ys_train,
            bs=ws_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            Σ=Σ,
            solver=SOLVER,
            warm_start=False,
        ).solve([body], must_realize=True)

        regression_results["params"].append(id_res.params[0])
        regression_results["solve_times"].append(id_res.solve_time)
        regression_results["num_iters"].append(id_res.iters)

        # verify realizability
        solved, verify_res = body.is_realizable(
            verbose=True, solver=SOLVER, warm_start=False
        )
        assert solved
        verification_results["solve_times"].append(verify_res.solve_time)
        verification_results["num_iters"].append(verify_res.iters)

    # save the results
    with open("bunny_verification.pkl", "wb") as f:
        pickle.dump(verification_results, f)
    with open("bunny_regression.pkl", "wb") as f:
        pickle.dump(regression_results, f)
    print("Saved results.")

    mean_func = np.median
    print("Identification")
    print(f"Solve time = {mean_func(regression_results['solve_times'])}")
    print(f"Iterations = {mean_func(regression_results['num_iters'])}")

    print("\nVerification")
    print(f"Solve time = {mean_func(verification_results['solve_times'])}")
    print(f"Iterations = {mean_func(verification_results['num_iters'])}")


main()
