"""Generate simulated data for random polyhedral rigid bodies."""
import argparse
from pathlib import Path
import pickle
import time

import numpy as np
import pybullet_data
import pinocchio
import cvxpy as cp

from mobile_manipulation_central.kinematics import RobotKinematics
import inertial_params as ip

import IPython


FREQ = 10
DURATION = 10
NUM_STEPS = DURATION * FREQ
TIMESTEP = 1.0 / FREQ

NUM_OBJ = 10
NUM_TRAJ = 2
NUM_PRIMITIVES = 100
BOUNDING_BOX_HALF_EXTENT = 0.5
OFFSET = np.array([0.2, 0, 0])
MASS = 1.0
GRAVITY = np.array([0, 0, -9.81])


def simulate_trajectories(q0, qds, timescaling, model, params):
    q = q0.copy()

    traj_idx = 0
    trajectory = ip.PointToPointTrajectory(
        q, delta=qds[traj_idx, :] - q, timescaling=timescaling
    )

    Vs = []
    As = []
    ws = []
    Ys = []

    Vs_noisy = []
    As_noisy = []
    ws_noisy = []
    Ys_noisy = []

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
        C_we = model.link_pose(rotation_matrix=True)[1]
        V = np.concatenate(model.link_velocity(frame="local"))
        A = np.concatenate(model.link_classical_acceleration(frame="local"))

        # account for gravity
        G = np.concatenate((C_we.T @ GRAVITY, np.zeros(3)))
        # G = np.zeros(6)
        A -= G
        w = params.body_wrench(V, A)

        Vs.append(V)
        As.append(A)
        ws.append(w)
        Ys.append(ip.body_regressor(V, A))

        # noisy version
        q_noisy = q + np.random.normal(scale=0.01, size=q.shape)
        v_noisy = v + np.random.normal(scale=0.01 / TIMESTEP, size=q.shape)
        a_noisy = a + np.random.normal(scale=0.01 / TIMESTEP**2, size=q.shape)

        model.forward(q_noisy, v_noisy, a_noisy)
        C_we_noisy = model.link_pose(rotation_matrix=True)[1]
        V_noisy = np.concatenate(model.link_velocity(frame="local"))
        A_noisy = np.concatenate(model.link_classical_acceleration(frame="local"))

        G_noisy = np.concatenate((C_we_noisy.T @ GRAVITY, np.zeros(3)))
        A_noisy -= G_noisy
        w_noisy = params.body_wrench(V_noisy, A_noisy)

        Vs_noisy.append(V_noisy)
        As_noisy.append(A_noisy)
        ws_noisy.append(w_noisy)
        Ys_noisy.append(ip.body_regressor(V_noisy, A_noisy))

    return {
        "Vs": Vs,
        "As": As,
        "ws": ws,
        "Ys": Ys,
        "Vs_noisy": Vs_noisy,
        "As_noisy": As_noisy,
        "ws_noisy": ws_noisy,
        "Ys_noisy": Ys_noisy,
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
    args = parser.parse_args()

    urdf_path = (ip.get_urdf_path() / "ur10.urdf").as_posix()
    model = RobotKinematics.from_urdf_file(urdf_path, tool_link_name="ur10_tool0")

    # urdf_path = Path(pybullet_data.getDataPath()) / "kuka_iiwa/model.urdf"
    # model = ip.RobotKinematics.from_urdf_file(
    #     urdf_path.as_posix(), tool_link_name="lbr_iiwa_link_7"
    # )

    # generate random point-to-point joint space trajectories
    # TODO we would like to make sure there are no collisions (cube with arm,
    # arm with arm, arm with ground, cube with ground)
    q0 = np.zeros(model.nq)
    qds = np.pi * (np.random.random((NUM_TRAJ, model.nq)) - 0.5)
    timescaling = ip.QuinticTimeScaling(duration=DURATION / NUM_TRAJ)

    bounding_box = ip.AxisAlignedBox.cube(BOUNDING_BOX_HALF_EXTENT, center=OFFSET)

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
        obj_datum = simulate_trajectories(q0, qds, timescaling, model, params)

        obj_data.append(obj_datum)
        param_data.append(params)
        vertices_data.append(vertices)

    data = {
        "num_obj": NUM_OBJ,
        "obj_data": obj_data,
        "params": param_data,
        "vertices": vertices_data,
    }

    with open(args.outfile, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {args.outfile}")


main()
