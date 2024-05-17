#!/usr/bin/env python3
import argparse
import pickle

import numpy as np
import cvxpy as cp
import tqdm

from xacrodoc import XacroDoc
import rigeo as rg

import IPython

NUM_TRIALS = 10
RECORDING_TIMESTEP = 0.1
GRAVITY = np.array([0, 0, -9.81])

TRAIN_TEST_SPLIT = 0.5

SHUFFLE = True
REGULARIZATION_COEFF = 0
PIM_EPS = 1e-4
SOLVER = cp.MOSEK

TORQUE_NOISE_COV = np.diag([0.1, 0.1, 0.1]) ** 2
TORQUE_NOISE_BIAS = np.array([0, 0, 0])


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


def compute_kinematic_state(t, offsets):
    q = np.sin(t + offsets)
    v = np.cos(t + offsets)
    a = -np.sin(t + offsets)
    return q, v, a


def compute_trajectory(multi):
    n, ts_eval = rg.compute_evaluation_times(
        duration=2 * np.pi, step=RECORDING_TIMESTEP
    )

    qs = []
    vs = []
    as_ = []
    τs = []

    offsets = np.random.random(multi.nv) * 2 * np.pi

    # generate the trajectory
    for i in range(n):
        q, v, a = compute_kinematic_state(i * RECORDING_TIMESTEP, offsets)
        τ = multi.compute_joint_torques(q, v, a)

        qs.append(q)
        vs.append(v)
        as_.append(a)
        τs.append(τ)

    qs = np.array(qs)
    vs = np.array(vs)
    as_ = np.array(as_)
    τs = np.array(τs)
    return qs, vs, as_, τs


def shapes_can_realize(shapes, params):
    assert len(shapes) == len(params)
    for shape, p in zip(shapes, params):
        if not shape.can_realize(p):
            return False
    return True


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "shape", help="Which link shape to use.", choices=["box", "cylinder"]
    )
    parser.add_argument("outfile", help="File to save the results to.")
    args = parser.parse_args()

    # compile the URDF
    xacro_doc = XacroDoc.from_package_file(
        package_name="rigeo",
        relative_path=f"urdf/threelink_{args.shape}.urdf.xacro",
    )

    # load Pinocchio model
    multi = rg.MultiBody.from_urdf_string(xacro_doc.to_urdf_string(), gravity=GRAVITY)

    joints = ["link1_joint", "link2_joint", "link3_joint"]
    bodies = multi.get_bodies(joints)
    bodies_mbe = [body.mbes() for body in bodies]

    params = [body.params for body in bodies]
    shapes = [body.shapes[0] for body in bodies]

    results = result_dict()

    for i in tqdm.tqdm(range(NUM_TRIALS)):
        # joint trajectory
        qs, vs, as_, τs = compute_trajectory(multi)
        n = qs.shape[0]

        # add noise
        torque_noise = np.random.multivariate_normal(
            mean=TORQUE_NOISE_BIAS, cov=TORQUE_NOISE_COV, size=τs.shape[0]
        )
        τs_noisy = τs + torque_noise

        Ys = []
        for i in range(n):
            Ys.append(multi.compute_joint_torque_regressor(qs[i], vs[i], as_[i]))
        Ys = np.array(Ys)

        # partition data
        idx = np.arange(n)
        if SHUFFLE:
            np.random.shuffle(idx)

        Ys = Ys[idx]
        τs = τs[idx]
        τs_noisy = τs_noisy[idx]

        n = vs.shape[0]
        n_train = int(TRAIN_TEST_SPLIT * n)

        Ys_train = Ys[:n_train]
        τs_train = τs_noisy[:n_train]

        Ys_test = Ys[n_train:]
        τs_test = τs[n_train:]

        res_nom = rg.IdentificationProblem(
            As=Ys_train,
            bs=τs_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            solver=SOLVER,
            warm_start=False,
        ).solve(bodies, must_realize=False)

        res_ell = rg.IdentificationProblem(
            As=Ys_train,
            bs=τs_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            solver=SOLVER,
            warm_start=False,
        ).solve(bodies_mbe, must_realize=True)

        res_ell_com = rg.IdentificationProblem(
            As=Ys_train,
            bs=τs_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            solver=SOLVER,
            warm_start=False,
        ).solve(bodies_mbe, must_realize=True, com_bounding_shapes=shapes)

        res_poly = rg.IdentificationProblem(
            As=Ys_train,
            bs=τs_train,
            γ=REGULARIZATION_COEFF,
            ε=PIM_EPS,
            solver=SOLVER,
            warm_start=False,
        ).solve(bodies, must_realize=True)

        params_nom = res_nom.params
        params_ell = res_ell.params
        params_ell_com = res_ell_com.params
        params_poly = res_poly.params

        # Riemannian errors
        results["nominal"]["riemannian_errors"].append(
            np.sum(
                [
                    rg.positive_definite_distance(p.J, pn.J)
                    for p, pn in zip(params, params_nom)
                ]
            )
        )
        results["ellipsoid"]["riemannian_errors"].append(
            np.sum(
                [
                    rg.positive_definite_distance(p.J, pn.J)
                    for p, pn in zip(params, params_ell)
                ]
            )
        )
        results["ell_com"]["riemannian_errors"].append(
            np.sum(
                [
                    rg.positive_definite_distance(p.J, pn.J)
                    for p, pn in zip(params, params_ell_com)
                ]
            )
        )
        results["polyhedron"]["riemannian_errors"].append(
            np.sum(
                [
                    rg.positive_definite_distance(p.J, pn.J)
                    for p, pn in zip(params, params_poly)
                ]
            )
        )

        # validation errors
        results["nominal"]["validation_errors"].append(
            rg.validation_rmse(
                Ys_test, τs_test, np.concatenate([p.vec for p in params_nom]), W=None
            )
        )
        results["ellipsoid"]["validation_errors"].append(
            rg.validation_rmse(
                Ys_test, τs_test, np.concatenate([p.vec for p in params_ell]), W=None
            )
        )
        results["ell_com"]["validation_errors"].append(
            rg.validation_rmse(
                Ys_test,
                τs_test,
                np.concatenate([p.vec for p in params_ell_com]),
                W=None,
            )
        )
        results["polyhedron"]["validation_errors"].append(
            rg.validation_rmse(
                Ys_test, τs_test, np.concatenate([p.vec for p in params_poly]), W=None
            )
        )

        # objective values
        results["nominal"]["objective_values"].append(res_nom.objective)
        results["ellipsoid"]["objective_values"].append(res_ell.objective)
        results["ell_com"]["objective_values"].append(res_ell_com.objective)
        results["polyhedron"]["objective_values"].append(res_poly.objective)

        # number of iterations
        results["nominal"]["num_iters"].append(res_nom.iters)
        results["ellipsoid"]["num_iters"].append(res_ell.iters)
        results["ell_com"]["num_iters"].append(res_ell_com.iters)
        results["polyhedron"]["num_iters"].append(res_poly.iters)

        # solve times
        results["nominal"]["solve_times"].append(res_nom.solve_time)
        results["ellipsoid"]["solve_times"].append(res_ell.solve_time)
        results["ell_com"]["solve_times"].append(res_ell_com.solve_time)
        results["polyhedron"]["solve_times"].append(res_poly.solve_time)

        try:
            if shapes_can_realize(shapes, params_nom):
                results["nominal"]["num_feasible"] += 1
        except cp.error.SolverError:
            # take failure to solve as infeasible
            pass

        try:
            if shapes_can_realize(shapes, params_ell):
                results["ellipsoid"]["num_feasible"] += 1
        except cp.error.SolverError:
            pass

        try:
            if shapes_can_realize(shapes, params_ell_com):
                results["ell_com"]["num_feasible"] += 1
        except cp.error.SolverError:
            pass

        assert shapes_can_realize(shapes, params_poly)
        results["polyhedron"]["num_feasible"] += 1

    # save the results
    with open(args.outfile, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {args.outfile}")


main()
