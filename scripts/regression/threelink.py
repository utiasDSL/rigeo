"""Generate simulated data for random polyhedral rigid bodies."""
import argparse
from pathlib import Path
import pickle
import time

import rospkg
import numpy as np
# import pinocchio
import pybullet as pyb
import pybullet_data
import pyb_utils

import mobile_manipulation_central as mm
import inertial_params as ip

import IPython

RECORDING_TIMESTEP = 0.1
SIM_FREQ = 100
GRAVITY = np.array([0, 0, -9.81])

VEL_NOISE_WIDTH = 0.1
VEL_NOISE_BIAS = 0

VISUALIZE = False


def sinusoidal_trajectory(t):
    b = np.array([0, 2 * np.pi, 4 * np.pi]) / 3
    q = np.array([np.sin(t + bi) for bi in b])
    v = np.array([np.cos(t + bi) for bi in b])
    a = np.array([-np.sin(t + bi) for bi in b])
    return q, v, a


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("outfile", help="File to save the data to.")
    # args = parser.parse_args()

    # compile the URDF
    xacro_doc = mm.XacroDoc.from_package_file(
        package_name="inertial_params",
        relative_path="urdf/threelink.urdf.xacro",
    )

    # load Pinocchio model
    model = ip.RobotModel.from_urdf_string(xacro_doc.to_urdf_string(), gravity=GRAVITY)

    n, ts_eval = ip.compute_evaluation_times(duration=2 * np.pi, step=RECORDING_TIMESTEP)

    qs = []
    vs = []
    as_ = []
    τs = []

    # generate the trajectory
    for i in range(n):
        q, v, a = sinusoidal_trajectory(i * RECORDING_TIMESTEP)
        τ = model.compute_torques(q, v, a)

        qs.append(q)
        vs.append(v)
        as_.append(a)
        τs.append(τ)

    qs = np.array(qs)
    vs = np.array(vs)
    as_ = np.array(as_)
    τs = np.array(τs)

    vs_noise_raw = np.random.random(size=vs.shape) - 0.5  # mean = 0, width = 1
    vs_noisy = vs + VEL_NOISE_WIDTH * vs_noise_raw + VEL_NOISE_BIAS

    as_mid = (vs_noisy[1:, :] - vs_noisy[:-1, :]) / RECORDING_TIMESTEP
    vs_mid = (vs_noisy[1:, :] + vs_noisy[:-1, :]) / 2
    qs_mid = (qs[1:, :] - qs[:-1, :]) / 2  # TODO?

    Ys = []
    Ys_noisy = []
    for i in range(n):
        Ys.append(model.compute_joint_torque_regressor(qs[i], vs[i], as_[i]))
        if i < n - 1:
            Ys_noisy.append(model.compute_joint_torque_regressor(qs_mid[i], vs_mid[i], as_mid[i]))

    Ys = np.array(Ys)
    Ys_noisy = np.array(Ys_noisy)

    # TODO solve the id problem

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

    # robot.reset_joint_configuration(q0)

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
