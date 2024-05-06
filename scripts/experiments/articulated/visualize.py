#!/usr/bin/env python3
import argparse
import time
from enum import Enum

import numpy as np
import cvxpy as cp

import pybullet as pyb
import pybullet_data
import pyb_utils

from xacrodoc import XacroDoc
import rigeo as rg

import IPython


def main():
    np.set_printoptions(suppress=True, precision=6)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "shape", help="Which link shape to use.", choices=["box", "cylinder"]
    )
    args = parser.parse_args()

    # compile the URDF
    xacro_doc = XacroDoc.from_package_file(
        package_name="rigeo",
        relative_path=f"urdf/threelink_{args.shape}.urdf.xacro",
    )

    pyb.connect(pyb.GUI)
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
    # robot.set_joint_friction_forces([0, 0, 0])

    IPython.embed()


main()
