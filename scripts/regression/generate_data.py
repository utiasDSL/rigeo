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
NUM_PRIMITIVES = 10
BOUNDING_BOX_HALF_EXTENT = 0.1
OFFSET = np.array([0, 0, 0.2])
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
        A = A - G
        w = params.body_wrench(V, A)

        Vs.append(V)
        As.append(A)
        ws.append(w)
        Ys.append(ip.body_regressor(V, A))
    return {"Vs": Vs, "As": As, "ws": ws, "Ys": Ys}


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

        # generate random point masses in a box of half length r centered at
        # `offset` from the EE
        if args.type == "points":
            points = bounding_box.random_points(NUM_PRIMITIVES)
            if NUM_PRIMITIVES > 3:
                # convex hull computation only works in non-degenerate cases
                vertices = ip.convex_hull(points)
            else:
                vertices = points
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
