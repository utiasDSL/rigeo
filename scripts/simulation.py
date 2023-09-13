from pathlib import Path
import time

import numpy as np
import pybullet as pyb
import pybullet_data
import pinocchio

import pyb_utils
import inertial_params as ip

import IPython


FREQ = 100
DURATION = 10
NUM_STEPS = DURATION * FREQ
TIMESTEP = 1.0 / FREQ

NUM_TRAJ = 2


def main():
    np.random.seed(0)

    pyb.connect(pyb.GUI)
    pyb.setTimeStep(TIMESTEP)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.setGravity(0, 0, -9.81)
    pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    # KUKA iiwa robot arm
    kuka_id = pyb.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    robot = pyb_utils.Robot(kuka_id, "lbr_iiwa_joint_7")
    pos, orn = robot.get_link_pose()

    # rigid body
    offset = np.array([0, 0, 0.2])
    body = pyb_utils.BulletBody.box(pos + offset, half_extents=(0.1, 0.1, 0.1), mass=1)

    # rigidly attach body to the EE
    con_id = pyb.createConstraint(
        body.uid,
        -1,
        robot.uid,
        robot.tool_idx,
        pyb.JOINT_FIXED,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=offset,
    )

    urdf_path = Path(pybullet_data.getDataPath()) / "kuka_iiwa/model.urdf"
    model = pinocchio.buildModelFromUrdf(urdf_path.as_posix())

    IPython.embed()
    return

    # generate a random point-to-point joint space trajectories
    # TODO we would like to make sure there are no collisions (cube with arm,
    # arm with arm, arm with ground, cube with ground)
    Kp = np.eye(robot.num_joints)
    qds = np.pi * (np.random.random((NUM_TRAJ, robot.num_joints)) - 0.5)
    timescaling = ip.QuinticTimeScaling(duration=DURATION / NUM_TRAJ)
    traj_idx = 0

    q, _ = robot.get_joint_states()
    trajectory = ip.PointToPointTrajectory(
        q, delta=qds[traj_idx, :] - q, timescaling=timescaling
    )

    # TODO don't really need the simulator here: we can just sample each time as needed
    for i in range(NUM_STEPS):
        t = i * TIMESTEP
        q, v, a = trajectory.sample(t)
        # TODO need to be careful to get the reference frames right
        # robot.forward(q, v, a)  # TODO
        # compute P, V, A
        # compute body-frame w

    pyb.stepSimulation()
    for i in range(NUM_STEPS):
        t = i * TIMESTEP
        q, _ = robot.get_joint_states()

        # trajectory-tracking joint velocity controller
        if trajectory.done(t):
            traj_idx += 1
            trajectory = ip.PointToPointTrajectory(
                q, delta=qds[traj_idx, :] - q, timescaling=timescaling
            )
        qd, vd, ad = trajectory.sample(t)
        cmd_vel = Kp @ (qd - q) + vd
        robot.command_velocity(cmd_vel)

        # get the constraint forces
        wrench = np.array(pyb.getConstraintState(con_id))
        print(wrench)

        pyb.stepSimulation()

    IPython.embed()


main()
