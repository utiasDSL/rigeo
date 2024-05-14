#!/usr/bin/env python3
"""Generate simulated data for random polyhedral rigid bodies."""
import argparse
import pickle

import numpy as np

import rigeo as rg

import IPython

# noise
VEL_NOISE_WIDTH = 0
VEL_NOISE_BIAS = 0

WRENCH_NOISE_COV = np.diag([1.2, 1.2, 0.5, 0.02, 0.02, 0.03]) ** 2
WRENCH_NOISE_BIAS = np.array([5, 5, 5, 0, 0, 0])


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("body_pickle", help="Pickle file containing rigid body data.")
    parser.add_argument("outfile", help="Pickle file to save the data to.")
    args = parser.parse_args()

    with open(args.body_pickle, "rb") as f:
        body_data = pickle.load(f)

    obj_data_full = []
    obj_data_planar = []

    for i in range(body_data["num_obj"]):
        params = body_data["params"][i]

        # note noise will be different in each, but this is fine if we only
        # use one for training
        full_traj = rg.generate_rigid_body_trajectory2(
            params=params,
            vel_noise_width=VEL_NOISE_WIDTH,
            vel_noise_bias=VEL_NOISE_BIAS,
            wrench_noise_cov=WRENCH_NOISE_COV,
            wrench_noise_bias=WRENCH_NOISE_BIAS,
            planar=False,
        )
        planar_traj = rg.generate_rigid_body_trajectory2(
            params=params,
            vel_noise_width=VEL_NOISE_WIDTH,
            vel_noise_bias=VEL_NOISE_BIAS,
            wrench_noise_cov=WRENCH_NOISE_COV,
            wrench_noise_bias=WRENCH_NOISE_BIAS,
            planar=True,
        )
        obj_data_full.append(full_traj)
        obj_data_planar.append(planar_traj)

    # copy all the body data and add the trajectory data
    data = body_data.copy()
    data["obj_data_full"] = obj_data_full
    data["obj_data_planar"] = obj_data_planar
    data["vel_noise_bias"] = VEL_NOISE_BIAS
    data["vel_noise_width"] = VEL_NOISE_WIDTH
    data["wrench_noise_bias"] = WRENCH_NOISE_BIAS
    data["wrench_noise_cov"] = WRENCH_NOISE_COV

    with open(args.outfile, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {args.outfile}")


main()
