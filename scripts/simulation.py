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

USE_ELLIPSOID_CONSTRAINT = True
USE_POLYHEDRON_CONSTRAINT = False


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


class IPIDProblem:
    """Inertial parameter identification optimization problem."""
    def __init__(self, Ys, ws, w_noise):
        self.A = np.vstack(Ys)
        self.b = np.concatenate(ws + w_noise)
        self.θ = cp.Variable(10)
        self.Jopt = cp.Variable((4, 4), PSD=True)

    def _solve(self, extra_constraints=None):
        objective = cp.Minimize(cp.sum_squares(self.A @ self.θ - self.b))

        constraints = J_vec_constraint(self.Jopt, self.θ)
        if extra_constraints is not None:
            constraints.extend(extra_constraints)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        return ip.RigidBody.from_vector(self.θ.value)

    def solve_nominal(self):
        return self._solve()

    def solve_ellipsoid(self, ellipsoid):
        extra_constraints = [cp.trace(ellipsoid.Q @ self.Jopt) >= 0]
        return self._solve(extra_constraints)

    def solve_polyhedron(self, vertices):
        nv = vertices.shape[0]
        mvs = cp.Variable(nv)
        Vs = np.array([np.outer(v, v) for v in vertices])
        extra_constraints = [
            mvs >= 0,
            cp.sum(mvs) == self.θ[0],
            mvs.T @ vertices == self.θ[1:4],
            self.Jopt[:3, :3] << cp.sum([m * V for m, V in zip(mvs, Vs)]),
        ]
        return self._solve(extra_constraints)


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
    params = ip.RigidBody.from_point_masses(masses=masses, points=points)

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
        w = params.body_wrench(V, A)

        Vs.append(V)
        As.append(A)
        ws.append(w)
        Ys.append(ip.body_regressor(V, A))

    # add noise to the force measurements
    w_noise = np.random.normal(scale=WRENCH_STDEV, size=(NUM_STEPS, 6))

    prob = IPIDProblem(Ys, ws, w_noise)
    params_nom = prob.solve_nominal()
    params_ell = prob.solve_ellipsoid(ellipsoid)
    params_poly = prob.solve_polyhedron(vertices)

    print(f"nom err  = {np.linalg.norm(params.θ - params_nom.θ)}")
    print(f"ell err  = {np.linalg.norm(params.θ - params_ell.θ)}")
    print(f"poly err = {np.linalg.norm(params.θ - params_poly.θ)}")

    IPython.embed()


main()
