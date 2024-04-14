#!/usr/bin/env python3
import argparse
import pickle

import wrlparser
import numpy as np

import rigeo as rg

import IPython


# number of random parameter sets to generate
NUM_PARAMS = 10

# number of random point masses per parameter set
NUM_POINTS_BOUNDS = [10, 25]

# desired length of bounding box diagonal
BB_DIAG_LEN = 1

MASS_BOUNDS = [0.1, 1.0]

# noise
VEL_NOISE_WIDTH = 0.05
VEL_NOISE_BIAS = 0.1

WRENCH_NOISE_WIDTH = 0
WRENCH_NOISE_BIAS = 0


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("wrlfile", help="WRL/VRML file to load.")
    parser.add_argument("pklfile", help="Pickle file to save the data to.")
    args = parser.parse_args()

    scene = rg.WRL.from_file_path(args.wrlfile, diaglen=BB_DIAG_LEN)

    obj_data_full = []
    obj_data_planar = []
    param_data = []

    mass_width = MASS_BOUNDS[1] - MASS_BOUNDS[0]
    assert mass_width >= 0

    for i in range(NUM_PARAMS):
        # random total mass
        mass = np.random.random() * mass_width + MASS_BOUNDS[0]

        # random number of points masses
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

        # note noise will be different in each, but this is fine if we only use
        # one for training
        full_traj = rg.generate_rigid_body_trajectory2(
            params=params,
            vel_noise_width=VEL_NOISE_WIDTH,
            vel_noise_bias=VEL_NOISE_BIAS,
            wrench_noise_width=WRENCH_NOISE_WIDTH,
            wrench_noise_bias=WRENCH_NOISE_BIAS,
            planar=False,
        )
        planar_traj = rg.generate_rigid_body_trajectory2(
            params=params,
            vel_noise_width=VEL_NOISE_WIDTH,
            vel_noise_bias=VEL_NOISE_BIAS,
            wrench_noise_width=WRENCH_NOISE_WIDTH,
            wrench_noise_bias=WRENCH_NOISE_BIAS,
            planar=True,
        )
        obj_data_full.append(full_traj)
        obj_data_planar.append(planar_traj)
        param_data.append(params)

    data = {
        "num_params": NUM_PARAMS,
        "obj_data_full": obj_data_full,
        "obj_data_planar": obj_data_planar,
        "params": param_data,
        "bb_diag_len": BB_DIAG_LEN,
    }

    with open(args.pklfile, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {args.pklfile}")


main()
