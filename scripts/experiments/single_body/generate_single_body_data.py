#!/usr/bin/env python3
"""Generate simulated data for random polyhedral rigid bodies."""
import argparse
import pickle

import numpy as np

import rigeo as rg

import IPython

# NOTE: IMPORTANT: scale wrench so that angular part is more reasonable


NUM_OBJ = 10
NUM_PRIMITIVE_BOUNDS = [10, 25]
BOUNDING_BOX_HALF_EXTENTS = [0.5, 0.5, 0.5]
MASS_BOUNDS = [0.1, 1.0]
# OFFSET = np.array([0.2, 0, 0])
OFFSET = np.array([0, 0, 0])

# noise
VEL_NOISE_WIDTH = 0.1
VEL_NOISE_BIAS = 0.1

WRENCH_NOISE_WIDTH = 0
WRENCH_NOISE_BIAS = 0


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help="Pickle file to save the data to.")
    parser.add_argument(
        "--type",
        choices=["points", "boxes"],
        help="Type of primitive to generate to make the random bodies.",
        default="points",
    )
    args = parser.parse_args()

    bounding_box = rg.Box(BOUNDING_BOX_HALF_EXTENTS, center=OFFSET)

    obj_data_full = []
    obj_data_planar = []
    param_data = []
    vertices_data = []

    mass_width = MASS_BOUNDS[1] - MASS_BOUNDS[0]
    assert mass_width >= 0

    for i in range(NUM_OBJ):
        # random total mass
        mass = np.random.random() * mass_width + MASS_BOUNDS[0]

        # random point masses
        num_primitives = np.random.randint(
            low=NUM_PRIMITIVE_BOUNDS[0], high=NUM_PRIMITIVE_BOUNDS[1] + 1
        )
        masses = np.random.random(num_primitives)
        masses = masses / sum(masses) * mass

        if args.type == "points":
            # random point mass system contained in the bounding box
            points = bounding_box.random_points(num_primitives)
            points = np.atleast_2d(points)
            vertices = rg.convex_hull(points)
            params = rg.InertialParameters.from_point_masses(
                masses=masses, points=points
            )
        elif args.type == "boxes":
            # generate random boxes inside a larger one by defining each box
            # using two vertices
            points = bounding_box.random_points((num_primitives, 2))

            # compute and sum up the inertial params for each box
            all_params = []
            all_vertices = []
            for j in range(num_primitives):
                box = rg.Box.from_two_vertices(points[j, 0, :], points[j, 1, :])
                all_vertices.append(box.vertices)
                params = box.uniform_density_params(masses[j])
                all_params.append(params)
            params = sum(all_params)
            vertices = rg.convex_hull(np.vstack(all_vertices))

        assert np.isclose(params.mass, mass)
        assert bounding_box.contains(params.com)

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
        vertices_data.append(vertices)

    data = {
        "num_obj": NUM_OBJ,
        "bounding_box": bounding_box,
        "obj_data_full": obj_data_full,
        "obj_data_planar": obj_data_planar,
        "params": param_data,
        "vertices": vertices_data,
    }

    with open(args.outfile, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {args.outfile}")


main()
