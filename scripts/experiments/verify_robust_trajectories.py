#!/usr/bin/env python3
"""Compare maximum constraint violations with different DRIP conditions."""
import argparse
import datetime
import glob
import os
from pathlib import Path
import time
import yaml

import numpy as np
import cvxpy as cp
import rigeo as rg
import tqdm

import upright_control as ctrl
import upright_core as core
import upright_robust as rob

import IPython


class RunData:
    """Summary of a single run."""

    def __init__(self):
        # timesteps
        self.times = []

        # times to solve a single verification problem
        self.verify_times_moment = []
        self.verify_times_box = []
        self.verify_times_ell = []

        # constraint violations using different DRIP conditions
        self.violations_moment = []
        self.violations_box = []
        self.violations_ell = []


def parse_run_dir(directory):
    """Parse npz and config path from a data directory of a single run.

    Returns (config_path, npz_path), as strings."""
    dir_path = Path(directory)

    config_paths = glob.glob(dir_path.as_posix() + "/*.yaml")
    assert len(config_paths) == 1, f"Found {len(config_paths)} config files."
    config_path = config_paths[0]

    npz_paths = glob.glob(dir_path.as_posix() + "/*.npz")
    assert len(npz_paths) == 1, f"Found {len(npz_paths)} npz files."
    npz_path = npz_paths[0]

    return config_path, npz_path


def max_or_init(field, value):
    if field is None:
        return value
    return max(field, value)


def setup_drip_problem(mass, com_box, drip_constraints):
    θ = cp.Variable(10)
    hY = cp.Parameter(10)

    J = rg.pim_must_equal_vec(θ)
    c = J[:3, 3] / mass  # CoM
    m = J[3, 3]  # mass

    objective = cp.Maximize(hY @ θ)
    constraints = (
        rg.pim_psd(J)
        + drip_constraints(J)
        + com_box.must_contain(c)
        + [m == mass]
    )
    problem = cp.Problem(objective, constraints)
    return problem, hY


def compute_run_data(directory):
    """Compute the bounds for a single run."""
    config_path, npz_path = parse_run_dir(directory)
    config = core.parsing.load_config(config_path)

    # use the nominal configuration to compute the bounds
    ctrl_config = config["controller"]
    ctrl_config["balancing"]["arrangement"] = "nominal"
    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    robot = model.robot

    # no approx_inertia because we want the actual realizable bounds
    objects, contacts = rob.parse_objects_and_contacts(
        ctrl_config,
        model=model,
        compute_bounds=True,
        approx_inertia=False,
    )

    obj0 = list(objects.values())[0]
    mass = obj0.body.mass
    com_box = obj0.com_box
    bounding_box = obj0.bounding_box
    bounding_ell = bounding_box.mbe()

    name = list(objects.keys())[0]
    contacts0 = [c for c in contacts if c.contact.object2_name == name]
    name_index = rob.compute_object_name_index([name])
    H = rob.compute_cwc_face_form(name_index, contacts0)

    # TODO swapped angular and linear components
    H2 = H.copy()
    H[:, :3] = H2[:, 3:]
    H[:, 3:] = H2[:, :3]

    # setup the DRIP optimization problems
    problem_moment, hY_moment = setup_drip_problem(
        mass, com_box, bounding_box.moment_sdp_constraints
    )
    problem_box, hY_box = setup_drip_problem(
        mass, com_box, bounding_box.moment_box_vertex_constraints
    )
    problem_ell, hY_ell = setup_drip_problem(
        mass, com_box, bounding_ell.moment_constraints
    )

    data = np.load(npz_path)
    ts = data["ts"]
    xds = data["xds"]

    run_data = RunData()

    for i in tqdm.trange(ts.shape[0]):
        if ts[i] > model.settings.mpc.time_horizon:
            break
        run_data.times.append(ts[i])

        # check the *planned* state, rather than the actual one
        xd = xds[i, :]
        robot.forward_xu(x=xd)

        C_we = robot.link_pose(rotation_matrix=True)[1]
        V_e = np.concatenate(robot.link_velocity(frame="local"))
        G_e = rob.body_gravity6(C_ew=C_we.T)
        A_e = np.concatenate(robot.link_spatial_acceleration(frame="local"))

        V = rg.SV(linear=V_e[:3], angular=V_e[3:])
        A = rg.SV(linear=A_e[:3] - G_e[:3], angular=A_e[3:] - G_e[3:])
        Y = rg.RigidBody.regressor(V=V, A=A)

        verify_times_moment = []
        verify_times_box = []
        verify_times_ell = []

        violations_moment = []
        violations_box = []
        violations_ell = []

        for h in H:
            # moment constraints
            hY_moment.value = h @ Y
            t0 = time.time()
            problem_moment.solve(solver=cp.MOSEK)
            t1 = time.time()
            assert problem_moment.status == "optimal"
            verify_times_moment.append(t1 - t0)
            violations_moment.append(problem_moment.value)

            # box constraints
            hY_box.value = h @ Y
            t0 = time.time()
            problem_box.solve(solver=cp.MOSEK)
            t1 = time.time()
            assert problem_box.status == "optimal"
            verify_times_box.append(t1 - t0)
            violations_box.append(problem_box.value)

            # ellipsoid constraints
            hY_ell.value = h @ Y
            t0 = time.time()
            problem_ell.solve(solver=cp.MOSEK)
            t1 = time.time()
            assert problem_ell.status == "optimal"
            verify_times_ell.append(t1 - t0)
            violations_ell.append(problem_ell.value)

        run_data.verify_times_moment.append(verify_times_moment)
        run_data.verify_times_box.append(verify_times_box)
        run_data.verify_times_ell.append(verify_times_ell)

        run_data.violations_moment.append(violations_moment)
        run_data.violations_box.append(violations_box)
        run_data.violations_ell.append(violations_ell)

    return run_data


def sort_dir_key(d):
    """Sort run data directories.

    Their names are of the form ``run_[number]_[other stuff]``. Note that the
    number if not padded with zeros to make a fixed width string.

    Directories are sorted in increasing value of ``[number]``.
    """
    name = Path(d).name
    return int(name.split("_")[1])


def main():
    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", help="Directory containing config and npz file."
    )
    args = parser.parse_args()

    # iterate through all directories
    dirs = glob.glob(args.directory + "/*/")
    dirs.sort(key=sort_dir_key)

    run_data = []
    for i, d in enumerate(dirs[:3]):
        print(Path(d).name)
        run_data.append(compute_run_data(d))

    basename = os.path.basename(args.directory)
    data_file_name = f"{basename}_drip_data.npz"
    np.savez(
        data_file_name,
        times=np.array([r.times for r in run_data]),
        verify_times_moment=np.array([r.verify_times_moment for r in run_data]),
        verify_times_box=np.array([r.verify_times_box for r in run_data]),
        verify_times_ell=np.array([r.verify_times_ell for r in run_data]),
        violations_moment=np.array([r.violations_moment for r in run_data]),
        violations_box=np.array([r.violations_box for r in run_data]),
        violations_ell=np.array([r.violations_ell for r in run_data]),
    )
    print(f"Dumped data to {data_file_name}")


if __name__ == "__main__":
    main()
