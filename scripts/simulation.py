"""Generate simulated data for a random polyhedral rigid body, corrupt with
noise, then try to regress the inertial parameters using differently
constrained least squares problems."""
from pathlib import Path
import time

import numpy as np
import pybullet_data
import pinocchio
import cvxpy as cp

import inertial_params as ip

import IPython


FREQ = 10
DURATION = 10
NUM_STEPS = DURATION * FREQ
TIMESTEP = 1.0 / FREQ

NUM_TRAJ = 2
NUM_POINTS = 10
R = 0.1
OFFSET = np.array([0, 0, 0.2])
MASS = 1.0
GRAVITY = np.array([0, 0, -9.81])

WRENCH_STDEV = 1.0

USE_ELLIPSOID_CONSTRAINT = False
USE_POLYHEDRON_CONSTRAINT = True


class RigidBody:
    """Inertial parameters of a rigid body.

    Parameters
    ----------
    mass : float
        Mass of the body.
    com : iterable
        Center of mass of the body w.r.t. to some reference point O.
    I : np.ndarray
        3x3 inertia matrix about w.r.t. O
    """

    def __init__(self, mass, com, I):
        self.mass = mass
        self.com = com
        self.I = I
        self.H = ip.I2H(I)
        self.J = ip.pseudo_inertia_matrix(mass, com, self.H)
        self.θ = np.concatenate([[mass], mass * com, ip.vech(self.I)])

        # TODO from_θ constructor
        # TODO from_mcH perhaps?

        S = ip.skew3(com)
        # self.I = Ic - mass * S @ S
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
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    # generate random point masses in a box of half length r centered at
    # `offset` from the EE
    points = R * (2 * np.random.random((NUM_POINTS, 3)) - 1) + OFFSET
    masses = np.random.random(NUM_POINTS)
    masses = masses / sum(masses) * MASS

    vertices = ip.convex_hull(points)
    ellipsoid = ip.minimum_bounding_ellipsoid(vertices)

    # ground truth
    com = ip.point_mass_system_com(masses, points)
    I = ip.point_mass_system_inertia(masses, points)[1]
    # TODO .from_point_masses(masses, points)
    body = RigidBody(mass=MASS, com=com, I=I)

    urdf_path = Path(pybullet_data.getDataPath()) / "kuka_iiwa/model.urdf"
    model = ip.RobotKinematics.from_urdf_file(
        urdf_path.as_posix(), tool_link_name="lbr_iiwa_joint_7"
    )

    # generate a random point-to-point joint space trajectories
    # TODO we would like to make sure there are no collisions (cube with arm,
    # arm with arm, arm with ground, cube with ground)
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

    # generate the data
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
    w_noise = np.random.normal(scale=WRENCH_STDEV, size=(NUM_STEPS, 6))

    # build data matrices
    A = np.vstack(Ys)
    b = np.concatenate(ws + w_noise)
    Vs = np.array([np.outer(v, v) for v in vertices])


    def J_vec_constraint(J, θ):
        H = J[:3, :3]
        I = cp.trace(H) * np.eye(3) - H
        return [
            J[3, 3] == θ[0],
            J[:3, 3] == θ[1:4],
            I[0, :3] == θ[4:7],
            I[1, 1:3] == θ[7:9],
            I[2, 2] == θ[9],
        ]

    θ = cp.Variable(10)
    Jopt = cp.Variable((4, 4), PSD=True)
    objective = cp.Minimize(cp.sum_squares(A @ θ - b))
    constraints = J_vec_constraint(Jopt, θ)

    if USE_ELLIPSOID_CONSTRAINT:
        constraints.append(cp.trace(ellipsoid.Q @ Jopt) >= 0)

    if USE_POLYHEDRON_CONSTRAINT:
        nv = vertices.shape[0]
        mvs = cp.Variable(nv)
        constraints.extend(
            [
                mvs >= 0,
                cp.sum(mvs) == θ[0],
                mvs.T @ vertices == θ[1:4],
                Jopt[:3, :3] << cp.sum([m * V for m, V in zip(mvs, Vs)]),
            ]
        )

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    mass_opt, com_opt, I_opt = unvec_params(θ.value)

    print(f"true mass = {MASS}")
    print(f"true com  = {com}")
    print(f"true I    = {I}")

    print(f"opt mass = {mass_opt}")
    print(f"opt com  = {com_opt}")
    print(f"opt I    = {I_opt}")

    print(f"residual = {np.linalg.norm(body.θ - θ.value)}")

    IPython.embed()


main()
