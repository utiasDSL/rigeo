"""Generate simulated data for random polyhedral rigid bodies."""
import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import pybullet_data
import pinocchio
import cvxpy as cp
from scipy.integrate import solve_ivp

import inertial_params as ip

import IPython


NUM_OBJ = 20
NUM_PRIMITIVES = 10
BOUNDING_BOX_HALF_EXTENTS = [0.5, 0.5, 0.5]
# OFFSET = np.array([0.2, 0, 0])
OFFSET = np.array([0, 0, 0])
MASS = 10.0
# MASS = 10.0


def compute_eval_times(duration, step=0.1):
    n = int(np.ceil(duration / step))
    times = step * np.arange(n)
    return n, times


def generate_trajectory(params, duration=2 * np.pi, eval_step=0.1, planar=False):
    M = params.M

    def wrench(t):
        return np.array(
            [
                np.sin(t),
                np.sin(t + np.pi / 3),
                np.sin(t + 2 * np.pi / 3) if not planar else 0,
                np.sin(t + np.pi) if not planar else 0,
                np.sin(t + 4 * np.pi / 3) if not planar else 0,
                np.sin(t + 5 * np.pi / 3),
            ]
        )

    def f(t, ξ):
        # solve Newton-Euler for acceleration
        return np.linalg.solve(M, wrench(t) - ip.skew6(ξ) @ M @ ξ)

    # integrate the trajectory
    n, t_eval = compute_eval_times(duration=duration, step=eval_step)
    res = solve_ivp(fun=f, t_span=[0, duration], y0=np.zeros(6), t_eval=t_eval)
    assert np.allclose(res.t, t_eval)

    # true trajectory
    velocities = res.y.T
    accelerations = np.array([f(t, ξ) for t, ξ in zip(t_eval, velocities)])
    wrenches = np.array([wrench(t) for t in t_eval])

    # apply noise to velocity
    vel_noise_raw = np.random.random(size=velocities.shape) - 0.5  # mean = 0, width = 1
    width = 0.1
    bias = 0.1
    # width = 0.0
    # bias = 0
    vel_noise = width * vel_noise_raw + bias
    if planar:
        vel_noise[:, 2:5] = 0
    velocities_noisy = velocities + vel_noise

    # compute midpoint values
    accelerations_mid = accelerations.copy()
    for i in range(1, n - 1):
        accelerations_mid[i] = (velocities_noisy[i + 1] - velocities_noisy[i - 1]) / (2 * eval_step)
    # accelerations_mid = (velocities_noisy[1:, :] - velocities_noisy[:-1, :]) / eval_step
    # velocities_mid = (velocities_noisy[1:, :] + velocities_noisy[:-1, :]) / 2
    velocities_mid = velocities_noisy  #[:-1, :]  # NOTE

    # wrenches_mid = (wrenches[1:, :] + wrenches[:-1, :]) / 2

    # apply noise to wrench
    wrench_noise_raw = np.random.random(size=wrenches.shape) - 0.5
    wrench_noise = 0 * wrench_noise_raw + 0
    wrenches_noisy = wrenches + wrench_noise

    return {
        "Vs": velocities,
        "As": accelerations,
        "ws": wrenches,
        "Vs_noisy": velocities_mid,
        "As_noisy": accelerations_mid,
        "ws_noisy": wrenches_noisy,
    }


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help="Pickle file to save the data to.")
    parser.add_argument(
        "--type",
        choices=["points", "boxes"],
        help="Type of primitive to generate to make the random bodies.",
        required=True,
    )
    parser.add_argument(
        "--planar",
        action="store_true",
        help="Only use a trajectory in the x-y plane.",
        required=False,
    )
    args = parser.parse_args()

    # bounding_box = ip.AxisAlignedBox.cube(BOUNDING_BOX_HALF_EXTENT, center=OFFSET)
    bounding_box = ip.AxisAlignedBox(BOUNDING_BOX_HALF_EXTENTS, center=OFFSET)

    obj_data = []
    param_data = []
    vertices_data = []

    for i in range(NUM_OBJ):
        masses = np.random.random(NUM_PRIMITIVES)
        masses = masses / sum(masses) * MASS

        if args.type == "points":
            # random point mass system contained in the bounding box
            points = bounding_box.random_points(NUM_PRIMITIVES)
            points = np.atleast_2d(points)
            vertices = ip.convex_hull(points)
            params = ip.RigidBody.from_point_masses(masses=masses, points=points)
        elif args.type == "boxes":
            # generate random boxes inside a larger one by defining each box
            # using two vertices
            points = bounding_box.random_points((NUM_PRIMITIVES, 2))

            # compute and sum up the inertial params for each box
            all_params = []
            all_vertices = []
            for j in range(NUM_PRIMITIVES):
                box = ip.AxisAlignedBox.from_two_vertices(
                    points[j, 0, :], points[j, 1, :]
                )
                all_vertices.append(box.vertices)
                Ic = ip.cuboid_inertia_matrix(masses[j], box.half_extents)
                Hc = ip.I2H(Ic)
                params = ip.RigidBody.translate_from_com(
                    mass=masses[j], h=masses[j] * box.center, Hc=Hc
                )
                all_params.append(params)
            params = sum(all_params)
            vertices = ip.convex_hull(np.vstack(all_vertices))

        assert np.isclose(params.mass, MASS)
        assert bounding_box.contains(params.com)

        obj_datum = generate_trajectory(params, planar=args.planar)

        obj_data.append(obj_datum)
        param_data.append(params)
        vertices_data.append(vertices)

    data = {
        "num_obj": NUM_OBJ,
        "bounding_box": bounding_box,
        "obj_data": obj_data,
        "params": param_data,
        "vertices": vertices_data,
    }

    with open(args.outfile, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {args.outfile}")


main()
