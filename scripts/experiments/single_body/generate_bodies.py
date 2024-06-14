#!/usr/bin/env python3
"""Generate simulated random polyhedral rigid bodies."""
import argparse
import pickle

import numpy as np

import rigeo as rg

NUM_OBJ = 1000
NUM_PRIMITIVE_BOUNDS = [10, 30]
BOUNDING_BOX_HALF_EXTENTS = [0.5, 0.5, 0.5]
MASS_BOUNDS = [0.5, 5.0]
OFFSET = np.array([0, 0, 0])


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

    bounding_box = rg.Box(half_extents=BOUNDING_BOX_HALF_EXTENTS, center=OFFSET)

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

        param_data.append(params)
        vertices_data.append(vertices)

    data = {
        "num_obj": NUM_OBJ,
        "num_primitive_bounds": NUM_PRIMITIVE_BOUNDS,
        "mass_bounds": MASS_BOUNDS,
        "bounding_box": bounding_box,
        "params": param_data,
        "vertices": vertices_data,
    }

    with open(args.outfile, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {args.outfile}")


main()
