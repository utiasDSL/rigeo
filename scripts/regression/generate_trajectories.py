"""Generate simulated data for random polyhedral rigid bodies."""
import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import pinocchio
import pybullet as pyb
import pybullet_data
import pyb_utils

import inertial_params as ip

import IPython


FREQ = 10
DURATION = 10
NUM_STEPS = DURATION * FREQ
TIMESTEP = 1.0 / FREQ

SIM_FREQ = 100

NUM_OBJ = 10
NUM_TRAJ = 2
NUM_PRIMITIVES = 10
BOUNDING_BOX_HALF_EXTENT = 0.1
OFFSET = np.array([0, 0, 0.2])
MASS = 1.0
GRAVITY = np.array([0, 0, -9.81])


# TODO we need to make this more sophisticated
def in_collision(model, q):
    elbow_idx = model.get_link_index("ur10_elbow_joint")
    wrist1_idx = model.get_link_index("ur10_wrist_1_joint")
    model.forward(q)

    for idx in [elbow_idx, wrist1_idx]:
        r = model.link_pose(link_idx=idx)[0]
        if r[2] < 0.2:
            return True
    return False


def generate_trajectory(model, q0, timescaling, max_tries=100, step=0.1):
    for _ in range(max_tries):
        # random goal in joint space
        q_goal = 2 * np.pi * (np.random.random(model.nq) - 0.5)

        # discretize the path
        delta = q_goal - q0
        length = np.linalg.norm(delta)
        n = int(length / step) + 1
        points = np.outer(np.linspace(0, 1, n), delta) + q0

        # check each point for collision
        for point in points:
            if in_collision(model, point):
                continue

        # if no collisions, build and return the trajectory
        trajectory = ip.PointToPointTrajectory(
            q0, delta=delta, timescaling=timescaling
        )
        return qd, trajectory

    raise Exception("Failed to generate collision free point!")


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help="File to save the data to.")
    args = parser.parse_args()

    urdf_path = (ip.get_urdf_path() / "ur10.urdf").as_posix()
    model = ip.RobotKinematics.from_urdf_file(urdf_path, tool_link_name="ur10_tool0")

    pyb.connect(pyb.GUI)
    pyb.setTimeStep(1.0 / SIM_FREQ)
    pyb.setGravity(*GRAVITY)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    robot_id = pyb.loadURDF(
        urdf_path,
        [0, 0, 0],
        useFixedBase=True,
    )
    robot = pyb_utils.Robot(robot_id, tool_joint_name="ur10_flange-tool0")
    # q0 = np.array([0, -0.1745, 1.5708, -1.3963, 1.5708, 1.3100])
    q0 = np.array([0, -np.pi / 2, 0, 0, 0, 0])
    robot.reset_joint_configuration(q0)

    model.forward(q0)
    r_ew_w, C_we = model.link_pose()
    r_ow_w = r_ew_w + C_we @ OFFSET
    pyb_utils.BulletBody.box(r_ow_w, half_extents=[BOUNDING_BOX_HALF_EXTENT] * 3)

    timescaling = ip.QuinticTimeScaling(duration=DURATION / NUM_TRAJ)
    traj_idx = 0
    qd, trajectory = generate_trajectory(model, q0, timescaling)
    qds = [qd]

    t = 0
    dt = 1.0 / SIM_FREQ
    Kp = np.eye(robot.num_joints)
    while t < DURATION:
        if trajectory.done(t):
            traj_idx += 1
            qd, trajectory = generate_trajectory(model, q, timescaling)
            qds.append(qd)
        qd, vd, _ = trajectory.sample(t)
        q, v = robot.get_joint_states()
        u = Kp @ (qd - q) + vd
        robot.command_velocity(u)
        pyb.stepSimulation()
        time.sleep(dt)

        t += dt

    IPython.embed()
    return

    # data = {
    #     "num_obj": NUM_OBJ,
    #     "obj_data": obj_data,
    #     "params": param_data,
    #     "vertices": vertices_data,
    # }
    #
    # with open(args.outfile, "wb") as f:
    #     pickle.dump(data, f)
    # print(f"Saved data to {args.outfile}")


main()
