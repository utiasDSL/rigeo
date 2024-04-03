#!/usr/bin/env python3
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
from scipy.spatial.transform import Rotation

import rigeo as rg

import IPython


NUM_OBJ = 10
NUM_PRIMITIVE_BOUNDS = [10, 25]
BOUNDING_BOX_HALF_EXTENTS = [0.5, 0.5, 0.5]
# OFFSET = np.array([0.2, 0, 0])
OFFSET = np.array([0, 0, 0])
MASS = 1.0  # TODO vary as well?

# noise
VEL_NOISE_WIDTH = 0.05
VEL_NOISE_BIAS = 0.1

WRENCH_NOISE_WIDTH = 0
WRENCH_NOISE_BIAS = 0


# TODO it would be cool have a Rotation class with a Jacobian method
def SO3_jacobian(axis_angle, eps=1e-4):
    angle = np.linalg.norm(axis_angle)
    if angle < eps:
        return np.eye(3)  # TODO
    axis = axis_angle / angle

    s = np.sin(angle)
    c = np.cos(angle)

    J = (
        (np.eye(3) - np.outer(axis, axis)) * s / angle
        + np.outer(axis, axis)
        + (1 - c) * rg.skew3(axis) / angle
    )
    return J


def generate_trajectory_old(params, duration=2 * np.pi, eval_step=0.1, planar=False):
    """Generate a random trajectory for a rigid body.

    Parameters
    ----------
    params : ip.InertialParameters
        The inertial parameters of the body.
    duration : float, positive
        The duration of the trajectory.
    eval_step : float, positive
        Interval at which to sample the trajectory.
    planar : bool
        Set to ``True`` to restrict the body to planar motion.

    Returns
    -------
    """
    M = params.M

    def wrench(t):
        """Body-frame wrench applied to the rigid body."""
        return np.array(
            [
                np.sin(t),
                np.sin(t + np.pi / 3),
                np.sin(t + 2 * np.pi / 3),
                np.sin(t + np.pi),
                np.sin(t + 4 * np.pi / 3),
                np.sin(t + 5 * np.pi / 3),
            ]
        )

    def f(t, x, debug=False):
        aa = x[:3]  # rotation (axis-angle)
        ξ = x[3:]  # generalized velocity

        # solve for axis-angle derivative in world-frame
        ω_body = ξ[3:]
        C_wb = Rotation.from_rotvec(aa).as_matrix()
        ω_world = C_wb @ ω_body
        J = SO3_jacobian(aa)
        aa_dot = np.linalg.solve(SO3_jacobian(aa), ω_world)

        # solve Newton-Euler for acceleration
        ξ_dot = np.linalg.solve(M, wrench(t) - rg.skew6(ξ) @ M @ ξ)
        if planar:
            ξ_dot[2:5] = 0

        x_dot = np.concatenate((aa_dot, ξ_dot))
        # if debug:
        #     print("debug")
        #     IPython.embed()
        return x_dot
        # solve Newton-Euler for acceleration
        # return np.linalg.solve(M, wrench(t) - rg.skew6(ξ) @ M @ ξ)

    # integrate the trajectory
    n, t_eval = rg.compute_evaluation_times(duration=duration, step=eval_step)
    res = solve_ivp(fun=f, t_span=[0, duration], y0=np.zeros(9), t_eval=t_eval)
    if planar:
        import IPython
        IPython.embed()
    assert np.allclose(res.t, t_eval)

    xs = res.y.T
    axis_angles = xs[:, :3]
    C_wbs = Rotation.from_rotvec(axis_angles).as_matrix()

    # true trajectory
    velocities = xs[:, 3:]
    accelerations = np.array([f(t, x)[3:] for t, x in zip(t_eval, xs)])
    # wrenches = np.array([wrench(t) for t in t_eval])
    wrenches = np.array(
        [M @ A + rg.skew6(V) @ M @ V for V, A in zip(velocities, accelerations)]
    )

    # apply noise to velocity
    vel_noise_raw = np.random.random(size=velocities.shape) - 0.5  # mean = 0, width = 1
    vel_noise = VEL_NOISE_WIDTH * vel_noise_raw + VEL_NOISE_BIAS
    if planar:
        vel_noise[:, 2:5] = 0
    velocities_noisy = velocities + vel_noise

    # compute midpoint values
    # accelerations_mid = accelerations.copy()
    # for i in range(1, n - 1):
    #     accelerations_mid[i] = (velocities_noisy[i + 1] - velocities_noisy[i - 1]) / (
    #         2 * eval_step
    #     )
    accelerations_mid = (velocities_noisy[1:, :] - velocities_noisy[:-1, :]) / eval_step
    velocities_mid = (velocities_noisy[1:, :] + velocities_noisy[:-1, :]) / 2
    # velocities_mid = velocities_noisy  # [:-1, :]  # NOTE

    # wrenches_mid = (wrenches[1:, :] + wrenches[:-1, :]) / 2

    # apply noise to wrench
    wrench_noise_raw = np.random.random(size=wrenches.shape) - 0.5
    wrench_noise = 0 * wrench_noise_raw + 0
    wrenches_noisy = wrenches + wrench_noise

    return {
        "Vs": velocities,
        "As": accelerations,
        "ws": wrenches,
        "C_wbs": C_wbs,
        "Vs_noisy": velocities_mid,
        "As_noisy": accelerations_mid,
        "ws_noisy": wrenches_noisy,
    }


def transform_wrench_from_com(w, c):
    f, τ = w[:3], w[3:]
    return np.concatenate((f, τ + np.cross(c, f)))


def transform_twist_to_com(V, c):
    v, ω = V[:3], V[3:]
    return np.concatenate((v - np.cross(c, ω), ω))


def planar_M(M):
    M_planar = np.zeros((3, 3))
    M_planar[:2, :2] = M[:2, :2]
    M_planar[:2, 2] = M[:2, 5]
    M_planar[2, :2] = M[5, :2]
    M_planar[2, 2] = M[5, 5]
    return M_planar


def generate_trajectory(params, duration=2 * np.pi, eval_step=0.1, planar=False):
    """Generate a random trajectory for a rigid body.

    Parameters
    ----------
    params : rg.InertialParameters
        The inertial parameters of the body.
    duration : float, positive
        The duration of the trajectory.
    eval_step : float, positive
        Interval at which to sample the trajectory.
    planar : bool
        Set to ``True`` to restrict the body to planar motion.

    Returns
    -------
    """
    M = params.M.copy()
    c = params.com

    # TODO
    # M_planar = M.copy()
    # M_planar[:, 2:5] = 0
    # M_planar[2:5, 0] = 0
    # M2 = planar_M(M)

    def wrench(t):
        """Body-frame wrench applied to the rigid body."""
        w = np.array(
            [
                np.sin(t),
                np.sin(t + np.pi / 3),
                np.sin(t + 2 * np.pi / 3),
                np.sin(t + np.pi),
                np.sin(t + 4 * np.pi / 3),
                np.sin(t + 5 * np.pi / 3),
            ]
        )
        return w

    def f(t, V):
        """Evaluate acceleration at time t given velocity V."""
        # solve Newton-Euler for acceleration
        # TODO this is currently failing in the planar case
        # w = transform_wrench_from_com(w=wrench(t), c=c)
        w = wrench(t)
        A = np.linalg.solve(M, w - rg.skew6(V) @ M @ V)
        # if planar:
        #     # planar means no linear motion in z, no angular in x and y
        #     A[2:5] = 0
        #     # A[2] = 0
        # print(V)
        # if planar:
        #     r = (w - rg.skew6(V) @ M_planar @ V)[2:5]
        #     A_planar = np.linalg.solve(M2, r)
        #     print(A_planar)
        #     A = np.zeros(6)
        #     A[:2] = A_planar[:2]
        #     A[5] = A_planar[2]
        return A

    # integrate the trajectory
    n, t_eval = rg.compute_evaluation_times(duration=duration, step=eval_step)
    V0 = np.zeros(6)
    res = solve_ivp(fun=f, t_span=[0, duration], y0=V0, t_eval=t_eval)
    assert res.success, "IVP failed to solve!"
    assert np.allclose(res.t, t_eval)

    # extract true trajectory values
    Vs = res.y.T
    As = np.array([f(t, V) for t, V in zip(t_eval, Vs)])
    ws = np.array([wrench(t) for t in t_eval])

    # TODO is this correct?
    if planar:
        # zero out non-planar components and find the wrenches that would
        # achieve that
        Vs[:, 2:5] = 0
        As[:, 2:5] = 0
        ws = np.array([M @ A + rg.skew6(V) @ M @ V for V, A, in zip(Vs, As)])

    # apply noise to velocity
    vel_noise_raw = np.random.random(size=Vs.shape) - 0.5  # mean = 0, width = 1
    vel_noise = VEL_NOISE_WIDTH * vel_noise_raw + VEL_NOISE_BIAS
    if planar:
        vel_noise[:, 2:5] = 0
    Vs_noisy = Vs + vel_noise

    # compute midpoint values
    As_mid = (Vs_noisy[1:, :] - Vs_noisy[:-1, :]) / eval_step
    Vs_mid = (Vs_noisy[1:, :] + Vs_noisy[:-1, :]) / 2

    # apply noise to wrench
    w_noise_raw = np.random.random(size=ws.shape) - 0.5
    w_noise = WRENCH_NOISE_WIDTH * w_noise_raw + WRENCH_NOISE_BIAS
    ws_noisy = ws + w_noise

    return {
        "Vs": Vs,
        "As": As,
        "ws": ws,
        "Vs_noisy": Vs_mid,
        "As_noisy": As_mid,
        "ws_noisy": ws_noisy,
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
        default="points",
    )
    args = parser.parse_args()

    bounding_box = rg.Box(BOUNDING_BOX_HALF_EXTENTS, center=OFFSET)

    obj_data_full = []
    obj_data_planar = []
    param_data = []
    vertices_data = []

    for i in range(NUM_OBJ):
        num_primitives = np.random.randint(
            low=NUM_PRIMITIVE_BOUNDS[0], high=NUM_PRIMITIVE_BOUNDS[1] + 1
        )
        masses = np.random.random(num_primitives)
        masses = masses / sum(masses) * MASS

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

        assert np.isclose(params.mass, MASS)
        assert bounding_box.contains(params.com)

        # note noise will be different in each, but this is fine if we only use
        # one for training
        obj_data_full.append(generate_trajectory(params, planar=False))
        obj_data_planar.append(generate_trajectory(params, planar=True))
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
