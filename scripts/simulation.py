from pathlib import Path
import time

import numpy as np
# import pybullet as pyb
import pybullet_data
import pinocchio
import cvxpy as cp

# import pyb_utils
import inertial_params as ip

import IPython


FREQ = 10
DURATION = 10
NUM_STEPS = DURATION * FREQ
TIMESTEP = 1.0 / FREQ

NUM_TRAJ = 2

GRAVITY = np.array([0, 0, -9.81])


class RigidBody:
    """Inertial parameters of a rigid body.

    Parameters
    ----------
    mass : float
        Mass of the body.
    com : iterable
        Center of mass of the body w.r.t. to some reference point O.
    Ic : np.ndarray
        3x3 inertia matrix about the CoM
    """
    def __init__(self, mass, com, Ic):
        self.mass = mass
        self.com = com

        # transfer inertia from CoM to reference point
        # TODO not sure if it will be better to pass in Ic or just I directly
        S = ip.skew3(com)
        self.I = Ic - mass * S @ S

        self.M = np.block([[mass * np.eye(3), -mass * S], [mass * S, self.I]])

    def body_wrench(self, V, A):
        """Compute the body-frame wrench about the reference point."""
        return self.M @ A + ip.skew6(V) @ self.M @ V

def body_regressor(V, A):
    """Compute regressor matrix Y given body frame velocity V and acceleration A.

    The regressor maps the inertial parameters to the body inertial wrench: w = Yθ.
    """
    return ip.lift6(A) + ip.skew6(V) @ ip.lift6(V)


def unvec_params(θ):
    mass = θ[0]
    com = θ[1:4] / mass
    # fmt: off
    I = np.array([
        [θ[4], θ[5], θ[6]],
        [θ[5], θ[7], θ[8]],
        [θ[6], θ[8], θ[9]]
    ])
    # fmt: on
    return mass, com, I


def main():
    np.random.seed(0)

    # pyb.connect(pyb.GUI)
    # pyb.setTimeStep(TIMESTEP)
    # pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    # pyb.setGravity(0, 0, -9.81)
    # pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    #
    # # KUKA iiwa robot arm
    # kuka_id = pyb.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
    # robot = pyb_utils.Robot(kuka_id, "lbr_iiwa_joint_7")
    # pos, orn = robot.get_link_pose()

    # rigid body
    # offset = np.array([0, 0, 0.2])
    # body = pyb_utils.BulletBody.box(pos + offset, half_extents=(0.1, 0.1, 0.1), mass=1)

    # rigidly attach body to the EE
    # con_id = pyb.createConstraint(
    #     body.uid,
    #     -1,
    #     robot.uid,
    #     robot.tool_idx,
    #     pyb.JOINT_FIXED,
    #     jointAxis=[0, 1, 0],
    #     parentFramePosition=[0, 0, 0],
    #     childFramePosition=offset,
    # )

    offset = np.array([0, 0, 0.2])
    body = RigidBody(mass=1.0, com=offset, Ic=0.1*np.eye(3))

    urdf_path = Path(pybullet_data.getDataPath()) / "kuka_iiwa/model.urdf"
    model = ip.RobotKinematics.from_urdf_file(urdf_path.as_posix(), tool_link_name="lbr_iiwa_joint_7")

    # generate a random point-to-point joint space trajectories
    # TODO we would like to make sure there are no collisions (cube with arm,
    # arm with arm, arm with ground, cube with ground)
    # Kp = np.eye(robot.num_joints)
    qds = np.pi * (np.random.random((NUM_TRAJ, model.nq)) - 0.5)
    timescaling = ip.QuinticTimeScaling(duration=DURATION / NUM_TRAJ)
    traj_idx = 0

    q = np.zeros(model.nq)
    trajectory = ip.PointToPointTrajectory(
        q, delta=qds[traj_idx, :] - q, timescaling=timescaling
    )

    Vs = []
    As = []
    ws = []
    Ys = []

    for i in range(NUM_STEPS):
        t = i * TIMESTEP

        # joint space
        if trajectory.done(t):
            traj_idx += 1
            trajectory = ip.PointToPointTrajectory(
                q, delta=qds[traj_idx, :] - q, timescaling=timescaling
            )
        q, v, a = trajectory.sample(t)

        # task space
        model.forward(q, v, a)
        C_we = model.link_pose()[1]
        V = np.concatenate(model.link_velocity(frame="local"))
        A = np.concatenate(model.link_classical_acceleration(frame="local"))

        # account for gravity
        G = np.concatenate((C_we.T @ GRAVITY, np.zeros(3)))
        A = A - G
        w = body.body_wrench(V, A)

        Vs.append(V)
        As.append(A)
        ws.append(w)
        Ys.append(body_regressor(V, A))

    # add noise to the force measurements
    stdev = 1.0
    w_noise = np.random.normal(scale=stdev, size=(NUM_STEPS, 6))

    # TODO implement the different constraint approaches
    θ = cp.Variable(10)
    A = np.vstack(Ys)
    b = np.concatenate(ws + w_noise)
    objective = cp.Minimize(cp.sum_squares(A @ θ - b))
    problem = cp.Problem(objective)
    problem.solve(solver=cp.MOSEK)

    mass_opt, com_opt, I_opt = unvec_params(θ.value)
    Ic_opt = I_opt + mass_opt * ip.skew3(com_opt) @ ip.skew3(com_opt)
    print(f"mass = {mass_opt}")
    print(f"com  = {com_opt}")
    print(f"Ic   = {Ic_opt}")

    IPython.embed()

    # pyb.stepSimulation()
    # for i in range(NUM_STEPS):
    #     t = i * TIMESTEP
    #     q, _ = robot.get_joint_states()
    #
    #     # trajectory-tracking joint velocity controller
    #     if trajectory.done(t):
    #         traj_idx += 1
    #         trajectory = ip.PointToPointTrajectory(
    #             q, delta=qds[traj_idx, :] - q, timescaling=timescaling
    #         )
    #     qd, vd, ad = trajectory.sample(t)
    #     cmd_vel = Kp @ (qd - q) + vd
    #     robot.command_velocity(cmd_vel)
    #
    #     # get the constraint forces
    #     wrench = np.array(pyb.getConstraintState(con_id))
    #     print(wrench)
    #
    #     pyb.stepSimulation()

    # IPython.embed()


main()
