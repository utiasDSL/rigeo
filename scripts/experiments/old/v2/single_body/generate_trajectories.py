#!/usr/bin/env python3
"""Generate simulated trajectory data for random polyhedral rigid bodies."""
import argparse
import pickle

import numpy as np

import rigeo as rg

# noise
VEL_NOISE_COV_DIAG = None
VEL_NOISE_BIAS = None

FORCE_NOISE_COV_DIAG = np.array([1.2, 1.2, 0.5]) ** 2
TORQUE_NOISE_COV_DIAG = np.array([0.02, 0.02, 0.03]) ** 2


def main():
    np.set_printoptions(suppress=True, precision=6)
    rng = np.random.default_rng(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "body_pickle", help="Pickle file containing rigid body data."
    )
    parser.add_argument("outfile", help="Pickle file to save the data to.")
    parser.add_argument("--force-bias", default=0, type=float, help="Bias force.")
    args = parser.parse_args()

    wrench_noise_bias = rg.SV(linear=args.force_bias * np.ones(3))
    wrench_noise_cov_diag = rg.SV(
        linear=FORCE_NOISE_COV_DIAG, angular=TORQUE_NOISE_COV_DIAG
    )

    with open(args.body_pickle, "rb") as f:
        body_data = pickle.load(f)

    obj_data_full = []
    obj_data_planar = []

    for i in range(body_data["num_obj"]):
        params = body_data["params"][i]

        # note noise will be different in each, but this is fine if we only
        # use one for training
        full_traj = rg.generate_rigid_body_trajectory(
            params=params,
            vel_noise_cov_diag=VEL_NOISE_COV_DIAG,
            vel_noise_bias=VEL_NOISE_BIAS,
            wrench_noise_cov_diag=wrench_noise_cov_diag,
            wrench_noise_bias=wrench_noise_bias,
            planar=False,
            rng=rng,
        )
        planar_traj = rg.generate_rigid_body_trajectory(
            params=params,
            vel_noise_cov_diag=VEL_NOISE_COV_DIAG,
            vel_noise_bias=VEL_NOISE_BIAS,
            wrench_noise_cov_diag=wrench_noise_cov_diag,
            wrench_noise_bias=wrench_noise_bias,
            planar=True,
            rng=rng,
        )
        obj_data_full.append(full_traj)
        obj_data_planar.append(planar_traj)

    # copy all the body data and add the trajectory data
    data = body_data.copy()
    data["obj_data_full"] = obj_data_full
    data["obj_data_planar"] = obj_data_planar
    data["vel_noise_bias"] = VEL_NOISE_BIAS
    data["vel_noise_cov_diag"] = VEL_NOISE_COV_DIAG
    data["wrench_noise_bias"] = wrench_noise_bias
    data["wrench_noise_cov_diag"] = wrench_noise_cov_diag

    with open(args.outfile, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {args.outfile}")


main()
