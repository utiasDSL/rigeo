from pathlib import Path

import numpy as np
import hppfcl
import pinocchio
from spatialmath.base import r2q

import inertial_params as ip


def _ref_frame_from_string(s):
    """Translate a string to pinocchio's ReferenceFrame enum value."""
    s = s.lower()
    if s == "local":
        return pinocchio.ReferenceFrame.LOCAL
    if s == "local_world_aligned":
        return pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
    if s == "world":
        return pinocchio.ReferenceFrame.WORLD
    raise ValueError(f"{s} is not a valid Pinocchio reference frame.")


class RobotModel:
    """Class representing the model of a robot."""

    def __init__(self, model, geom_model, tool_link_name=None, gravity=None):
        self.model = model
        self.nq = model.nq
        self.nv = model.nv
        self.data = self.model.createData()

        self.geom_model = geom_model
        self.geom_data = geom_model.createData()

        self.tool_link_name = tool_link_name
        if tool_link_name is not None:
            self.tool_idx = self.get_link_index(tool_link_name)
        else:
            self.tool_idx = None

        if gravity is not None:
            gravity = np.array(gravity)
            assert gravity.shape == (3,)
            self.model.gravity.linear = gravity

        # build the boxes composing the robot
        self.boxes = {}
        for i in range(geom_model.ngeoms):
            geom = geom_model.geometryObjects[i]
            if geom.geometry.getNodeType() != hppfcl.hppfcl.NODE_TYPE.GEOM_BOX:
                continue

            center = geom.placement.translation
            assert np.allclose(geom.placement.rotation, np.eye(3))
            half_extents = geom.geometry.halfSide
            box = ip.Box(half_extents=half_extents, center=center)

            # remove trailing number from the geometry names
            name = "_".join(geom.name.split("_")[:-1])
            if name in self.boxes:
                raise ValueError("Box named {name} already exists!")
            self.boxes[name] = box

    @classmethod
    def from_urdf_string(cls, urdf_str, root_joint=None, tool_link_name=None, gravity=None):
        """Load the model from a URDF represented as a string."""
        if root_joint is not None:
            model = pinocchio.buildModelFromXML(urdf_str, root_joint)
        else:
            model = pinocchio.buildModelFromXML(urdf_str)
        geom_model = pinocchio.buildGeomFromUrdfString(model, urdf_str, pinocchio.COLLISION)
        return cls(model, geom_model, tool_link_name, gravity=gravity)

    # @classmethod
    # def from_urdf_file(cls, urdf_file_path, root_joint=None, tool_link_name=None, gravity=None):
    #     """Load the model directly from a URDF file."""
    #     with open(urdf_file_path) as f:
    #         urdf_str = f.read()
    #     return cls.from_urdf_string(urdf_str, root_joint, tool_link_name, gravity=gravity)

    def get_link_index(self, link_name):
        """Get index of a link by name."""
        # TODO: it would probably be desirable to rename "link" to "frame"
        # everywhere in this class
        if not self.model.existFrame(link_name):
            raise ValueError(f"Model has no frame named {link_name}.")
        return self.model.getFrameId(link_name)

    def forward(self, q, v=None, a=None):
        """Forward kinematics using (q, v, a) all in the world frame (i.e.,
        corresponding directly to the Pinocchio model."""
        if v is None:
            v = np.zeros(self.nv)
        if a is None:
            a = np.zeros(self.nv)

        assert q.shape == (self.nq,)
        assert v.shape == (self.nv,)
        assert a.shape == (self.nv,)

        pinocchio.forwardKinematics(self.model, self.data, q, v, a)
        pinocchio.updateFramePlacements(self.model, self.data)

    def compute_torques(self, q, v, a):
        """Compute the joint torques corresponding to a given motion.

        This takes ``model.gravity`` into account.

        Parameters
        ----------
        q : np.ndarray, shape (self.nq,)
            The joint positions.
        v : np.ndarray, shape (self.nv,)
            The joint velocities.
        a : np.ndarray, shape (self.nv,)
            The joint accelerations.

        Returns
        -------
        : np.ndarray, shape (self.nv,)
            The corresponding joint torques.
        """
        return pinocchio.rnea(self.model, self.data, q, v, a)

    def compute_joint_torque_regressor(self, q, v, a):
        Y_pin = pinocchio.computeJointTorqueRegressor(self.model, self.data, q, v, a)
        # pinocchio stores inertial parameter vector as
        #   θ = [m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz],
        # but I prefer to store them
        #   θ = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        # (i.e, Ixz and Iyy are swapped)
        Y = Y_pin.copy()
        for i in range(self.model.nv):
            Y[:, i * 10 + 6] = Y_pin[:, i * 10 + 7]
            Y[:, i * 10 + 7] = Y_pin[:, i * 10 + 6]
        return Y

    def link_pose(self, link_idx=None, rotation_matrix=False):
        """Get pose of link at index link_idx.

        Must call forward(q, ...) first.

        Returns a tuple (position, orientation). If `rotation_matrix` is True,
        then the orientation is a 3x3 matrix, otherwise it is a quaternion with
        the scalar part as the last element.
        """
        if link_idx is None:
            link_idx = self.tool_idx
        pose = self.data.oMf[link_idx]
        pos = pose.translation.copy()
        orn = pose.rotation.copy()
        if not rotation_matrix:
            orn = r2q(orn, order="xyzs")
        return pos, orn

    def link_velocity(self, link_idx=None, frame="local_world_aligned"):
        """Get velocity of link at index link_idx"""
        if link_idx is None:
            link_idx = self.tool_idx
        V = pinocchio.getFrameVelocity(
            self.model,
            self.data,
            link_idx,
            _ref_frame_from_string(frame),
        )
        return V.linear, V.angular

    def link_classical_acceleration(self, link_idx=None, frame="local_world_aligned"):
        """Get the classical acceleration of a link."""
        if link_idx is None:
            link_idx = self.tool_idx
        A = pinocchio.getFrameClassicalAcceleration(
            self.model,
            self.data,
            link_idx,
            _ref_frame_from_string(frame),
        )
        return A.linear, A.angular

    def link_spatial_acceleration(self, link_idx=None, frame="local_world_aligned"):
        """Get the spatial acceleration of a link."""
        if link_idx is None:
            link_idx = self.tool_idx
        A = pinocchio.getFrameAcceleration(
            self.model,
            self.data,
            link_idx,
            _ref_frame_from_string(frame),
        )
        return A.linear, A.angular

    def jacobian(self, q, link_idx=None, frame="local_world_aligned"):
        """Compute the robot geometric Jacobian."""
        if link_idx is None:
            link_idx = self.tool_idx
        return pinocchio.computeFrameJacobian(
            self.model,
            self.data,
            q,
            link_idx,
            _ref_frame_from_string(frame),
        )

