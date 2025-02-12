#!/usr/bin/env python3
import argparse
import pickle

import wrlparser
import numpy as np

import rigeo as rg


# number of random parameter sets to generate
NUM_PARAMS = 10

# number of random point masses per parameter set
NUM_POINTS_BOUNDS = [10, 30]

# desired length of bounding box diagonal
BB_DIAG_LEN = 1

MASS_BOUNDS = [0.5, 5.0]

# noise
VEL_NOISE_WIDTH = 0
VEL_NOISE_BIAS = 0

WRENCH_NOISE_COV = np.diag([1.2, 1.2, 0.5, 0.02, 0.02, 0.03]) ** 2
WRENCH_NOISE_BIAS = np.array([0, 0, 0, 0, 0, 0])


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("wrlfile", help="WRL/VRML file to load.")
    parser.add_argument("pklfile", help="Pickle file to save the data to.")
    args = parser.parse_args()

    scene = rg.WRL.from_file_path(args.wrlfile, diaglen=BB_DIAG_LEN)

    obj_data_full = []
    param_data = []

    mass_width = MASS_BOUNDS[1] - MASS_BOUNDS[0]
    assert mass_width >= 0

    for i in range(NUM_PARAMS):
        # random total mass
        mass = np.random.random() * mass_width + MASS_BOUNDS[0]

        # random number of point masses
        num_points = np.random.randint(
            low=NUM_POINTS_BOUNDS[0], high=NUM_POINTS_BOUNDS[1] + 1
        )

        # random masses
        masses = np.random.random(num_points)
        masses = masses / sum(masses) * mass

        # random points
        points = scene.random_points(num_points)
        params = rg.InertialParameters.from_point_masses(masses=masses, points=points)

        assert np.isclose(params.mass, mass)

        full_traj = rg.generate_rigid_body_trajectory2(
            params=params,
            vel_noise_width=VEL_NOISE_WIDTH,
            vel_noise_bias=VEL_NOISE_BIAS,
            wrench_noise_cov=WRENCH_NOISE_COV,
            wrench_noise_bias=WRENCH_NOISE_BIAS,
            planar=False,
        )
        obj_data_full.append(full_traj)
        param_data.append(params)

    data = {
        "num_params": NUM_PARAMS,
        "obj_data_full": obj_data_full,
        "params": param_data,
        "bb_diag_len": BB_DIAG_LEN,
        "wrench_noise_cov": WRENCH_NOISE_COV,
        "wrench_noise_bias": WRENCH_NOISE_BIAS,
    }

    with open(args.pklfile, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {args.pklfile}")


main()
