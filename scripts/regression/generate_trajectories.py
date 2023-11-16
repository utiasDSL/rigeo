"""Generate simulated data for random polyhedral rigid bodies."""
import argparse
from pathlib import Path
import pickle
import time

import rospkg
import numpy as np
import pinocchio
import pybullet as pyb
import pybullet_data
import pyb_utils

import mobile_manipulation_central as mm
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
    elbow_idx = model.get_link_index("ur10_arm_elbow_joint")
    wrist1_idx = model.get_link_index("ur10_arm_wrist_1_joint")
    model.forward(q)

    for idx in [elbow_idx, wrist1_idx]:
        r = model.link_pose(link_idx=idx)[0]
        if r[2] < 0.2:
            return True
    return False


def generate_arm_trajectory(model, q0_full, timescaling, max_tries=100, step=0.1):
    q0_base = q0_full[:3]
    q0_arm = q0_full[3:]

    for _ in range(max_tries):
        # random goal in joint space
        q_goal_arm = 2 * np.pi * (np.random.random(6) - 0.5)
        q_goal_full = np.concatenate((q0_base, q_goal_arm))

        # discretize the path
        delta_full = q_goal_full - q0_full
        distance = np.linalg.norm(delta_full)
        n = int(distance / step) + 1
        points = np.outer(np.linspace(0, 1, n), delta_full) + q0_full

        # check each point for collision
        for point in points:
            if in_collision(model, point):
                continue

        # if no collisions, build and return the trajectory
        delta_arm = q_goal_arm - q0_arm
        trajectory = ip.PointToPointTrajectory(q0_arm, delta=delta_arm, timescaling=timescaling)
        return q_goal_arm, trajectory

    raise Exception("Failed to generate collision free point!")


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help="File to save the data to.")
    args = parser.parse_args()

    model = mm.MobileManipulatorKinematics(tool_link_name="gripper")

    pyb.connect(pyb.GUI)
    pyb.setTimeStep(1.0 / SIM_FREQ)
    pyb.setGravity(*GRAVITY)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    xacro_doc = mm.XacroDoc.from_package_file(
        package_name="mobile_manipulation_central",
        relative_path="urdf/xacro/thing_pyb.urdf.xacro",
    )
    with xacro_doc.temp_urdf_file_path() as urdf_path:
        robot_id = pyb.loadURDF(
            urdf_path,
            [0, 0, 0],
            useFixedBase=True,
        )
    robot = pyb_utils.Robot(robot_id, tool_link_name="gripper")

    q0_full = np.array([0, 0, 0, 0, -np.pi / 2, 0, 0, 0, 0])
    robot.reset_joint_configuration(q0_full)
    model.forward(q0_full)

    # TODO we need to rigidly attach to the EE
    r_ew_w, C_we = model.link_pose(rotation_matrix=True)
    r_ow_w = r_ew_w + C_we @ OFFSET
    pyb_utils.BulletBody.box(r_ow_w, half_extents=[BOUNDING_BOX_HALF_EXTENT] * 3)

    # check that models are aligned
    # r_pyb, C_pyb = robot.get_link_frame_pose(as_rotation_matrix=True)
    # assert np.allclose(r_pyb, r_ew_w)
    # assert np.allclose(C_pyb, C_we)

    q0_arm = q0_full[3:]
    assert q0_arm.shape == (6,)
    timescaling = ip.QuinticTimeScaling(duration=DURATION / NUM_TRAJ)
    traj_idx = 0
    q_goal, trajectory = generate_arm_trajectory(model, q0_full, timescaling)
    q_goals = [q_goal]

    dt = 1.0 / SIM_FREQ
    Kp = np.eye(6)
    q_arm = q0_arm
    t = 0
    while t < DURATION:
        if trajectory.done(t):
            traj_idx += 1
            q_goal, trajectory = generate_arm_trajectory(model, q_arm, timescaling)
            q_goals.append(q_goal)

        q, v = robot.get_joint_states()
        q_arm = q[3:]
        v_arm = v[3:]

        qd_arm, vd_arm, _ = trajectory.sample(t)
        u_arm = Kp @ (qd_arm - q_arm) + vd_arm
        u = np.concatenate((np.zeros(3), u_arm))

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
