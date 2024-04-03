#!/usr/bin/env python3
import argparse
import time
from enum import Enum

import numpy as np
import cvxpy as cp

import pybullet as pyb
import pybullet_data
import pyb_utils

from xacrodoc import XacroDoc
import rigeo as rg

import IPython

# NOTE: the difference between poly and ellipsoid constraints depends on the
# seed and the regularization coefficient (decreasing this makes poly better
# when a poly is the actual bounding object)
RECORDING_TIMESTEP = 0.1
SIM_FREQ = 100
GRAVITY = np.array([0, 0, -9.81])

TRAIN_TEST_SPLIT = 0.5
VEL_NOISE_WIDTH = 0.1
VEL_NOISE_BIAS = 0

REGULARIZATION_COEFF = 1e-3
PIM_EPS = 1e-4
SOLVER = cp.MOSEK

SEED = 1
VISUALIZE = False


def sinusoidal_trajectory(t):
    b = np.array([0, 2 * np.pi, 4 * np.pi]) / 3
    q = np.sin(t + b)
    v = np.cos(t + b)
    a = -np.sin(t + b)
    return q, v, a


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(SEED)

    # compile the URDF
    xacro_doc = XacroDoc.from_package_file(
        package_name="rigeo",
        relative_path="urdf/threelink.urdf.xacro",
    )

    # load Pinocchio model
    multi = rg.MultiBody.from_urdf_string(xacro_doc.to_urdf_string(), gravity=GRAVITY)

    joints = ["link1_joint", "link2_joint", "link3_joint"]
    bodies = multi.get_bodies(joints)
    bodies_mbe = [body.mbes() for body in bodies]

    params = [body.params for body in bodies]
    θ = np.concatenate([p.vec for p in params])
    #
    # # TODO brittle
    boxes = [body.shapes[0] for body in bodies]
    ellipsoids = [box.mbe() for box in boxes]

    n, ts_eval = rg.compute_evaluation_times(
        duration=2 * np.pi, step=RECORDING_TIMESTEP
    )

    qs = []
    vs = []
    as_ = []
    τs = []

    # generate the trajectory
    for i in range(n):
        q, v, a = sinusoidal_trajectory(i * RECORDING_TIMESTEP)
        τ = multi.compute_joint_torques(q, v, a)

        qs.append(q)
        vs.append(v)
        as_.append(a)
        τs.append(τ)

    qs = np.array(qs)
    vs = np.array(vs)
    as_ = np.array(as_)
    τs = np.array(τs)

    # add noise
    vs_noise_raw = np.random.random(size=vs.shape) - 0.5  # mean = 0, width = 1
    vs_noisy = vs + VEL_NOISE_WIDTH * vs_noise_raw + VEL_NOISE_BIAS

    as_mid = (vs_noisy[1:, :] - vs_noisy[:-1, :]) / RECORDING_TIMESTEP
    vs_mid = (vs_noisy[1:, :] + vs_noisy[:-1, :]) / 2
    qs_mid = (qs[1:, :] - qs[:-1, :]) / 2  # TODO?

    Ys = []
    Ys_noisy = []
    for i in range(n):
        Ys.append(multi.compute_joint_torque_regressor(qs[i], vs[i], as_[i]))
        if i < n - 1:
            Ys_noisy.append(
                multi.compute_joint_torque_regressor(qs_mid[i], vs_mid[i], as_mid[i])
            )

    Ys = np.array(Ys)
    Ys_noisy = np.array(Ys_noisy)

    # partition data
    n = vs.shape[0]
    n_train = int(TRAIN_TEST_SPLIT * n)

    Ys_train = Ys_noisy[:n_train]
    τs_train = τs[:n_train]

    Ys_test = Ys[n_train:]
    τs_test = τs[n_train:]

    # solve the ID problem
    # prob = IPIDProblem(
    #     Ys_train, τs_train, reg_params=params, reg_coeff=REGULARIZATION_COEFF
    # )
    # params_nom = prob.solve(name="nominal")
    # params_poly = prob.solve(shapes=boxes, name="poly")
    # params_ell = prob.solve(shapes=ellipsoids, name="ellipsoid")


    prob = rg.IdentificationProblem(
        As=Ys_train, bs=τs_train, γ=REGULARIZATION_COEFF, ε=PIM_EPS, solver=SOLVER
    )

    def solve_with_info(bodies, must_realize, name):
        P = prob.solve(bodies, must_realize=must_realize)
        print(f"{name} iters = {prob.problem.solver_stats.num_iters}")
        print(f"{name} solve time = {prob.problem.solver_stats.solve_time}")
        print(f"{name} value = {prob.problem.value}")
        return P

    params_nom = solve_with_info(bodies, must_realize=False, name="nominal")
    params_poly = solve_with_info(bodies, must_realize=True, name="poly")
    params_ell = solve_with_info(bodies_mbe, must_realize=True, name="ellipsoid")

    # results
    riemannian_err_nom = np.sum(
        [rg.positive_definite_distance(p.J, pn.J) for p, pn in zip(params, params_nom)]
    )
    riemannian_err_ell = np.sum(
        [rg.positive_definite_distance(p.J, pn.J) for p, pn in zip(params, params_ell)]
    )
    riemannian_err_poly = np.sum(
        [rg.positive_definite_distance(p.J, pn.J) for p, pn in zip(params, params_poly)]
    )

    θ_nom = np.concatenate([p.vec for p in params_nom])
    validation_err_nom = rg.validation_rmse(Ys_test, τs_test, θ_nom)

    θ_ell = np.concatenate([p.vec for p in params_ell])
    validation_err_ell = rg.validation_rmse(Ys_test, τs_test, θ_ell)

    θ_poly = np.concatenate([p.vec for p in params_poly])
    validation_err_poly = rg.validation_rmse(Ys_test, τs_test, θ_poly)

    print("\nRiemannian error")
    print(f"nominal    = {riemannian_err_nom}")
    print(f"ellipsoid  = {riemannian_err_ell}")
    print(f"polyhedron = {riemannian_err_poly}")

    print("\nValidation error")
    print(f"nominal    = {validation_err_nom}")
    print(f"ellipsoid  = {validation_err_ell}")
    print(f"polyhedron = {validation_err_poly}")

    IPython.embed()

    if not VISUALIZE:
        return

    pyb.connect(pyb.GUI)
    pyb.setTimeStep(1.0 / SIM_FREQ)
    pyb.setGravity(*GRAVITY)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    # load PyBullet model
    with xacro_doc.temp_urdf_file_path() as urdf_path:
        robot_id = pyb.loadURDF(
            urdf_path,
            [0, 0, 0],
            useFixedBase=True,
        )
    robot = pyb_utils.Robot(robot_id, tool_link_name="link3")

    # remove joint friction (only relevant for torque control)
    robot.set_joint_friction_forces([0, 0, 0])

    q0 = sinusoidal_trajectory(0)[0]
    robot.reset_joint_configuration(q0)

    Kq = np.eye(model.nq)
    Kv = np.eye(model.nv)
    dt = 1.0 / SIM_FREQ
    t = 0
    while t < 2 * np.pi:
        qd, vd, ad = sinusoidal_trajectory(t)
        q, v = robot.get_joint_states()

        # computed torque control law
        α = ad + Kq @ (qd - q) + Kv @ (vd - v)
        u = model.compute_torques(q, v, α)

        robot.command_effort(u)
        # u = Kq @ (qd - q) + vd
        # robot.command_velocity(u)

        pyb.stepSimulation()
        time.sleep(dt)

        t += dt

    IPython.embed()
    return


main()
