from pathlib import Path

import numpy as np
import hppfcl
import pinocchio
from spatialmath.base import r2q

import inertial_params as ip


# TODO want something more elegant than this
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
        self.data = self.model.createData()

        self.nq = model.nq  # number of joint positions
        self.nv = model.nv  # number of joint velocities

        # tool
        self.tool_link_name = tool_link_name
        if tool_link_name is not None:
            self.tool_idx = self.get_frame_index(tool_link_name)
        else:
            self.tool_idx = None

        # gravity
        if gravity is not None:
            gravity = np.array(gravity)
            assert gravity.shape == (3,)
            self.model.gravity.linear = gravity

        # geometric model
        self.geom_model = geom_model
        self.geom_data = geom_model.createData()

        # build the boxes composing the robot
        # TODO basically we'd like to seamlessly parse this into rigeo's format
        # for all supported shapes
        self.boxes = {}
        for i in range(geom_model.ngeoms):
            geom = geom_model.geometryObjects[i]
            if geom.geometry.getNodeType() != hppfcl.hppfcl.NODE_TYPE.GEOM_BOX:
                continue

            half_extents = geom.geometry.halfSide
            center = geom.placement.translation
            rotation = geom.placement.rotation
            box = ip.Box(half_extents=half_extents, center=center, rotation=rotation)

            # remove trailing number from the geometry names
            name = "_".join(geom.name.split("_")[:-1])
            if name in self.boxes:
                raise ValueError("Box named {name} already exists!")
            self.boxes[name] = box

    @classmethod
    def from_urdf_string(
        cls, urdf_str, root_joint=None, tool_link_name=None, gravity=None
    ):
        """Load the model from a URDF string.

        Parameters
        ----------
        urdf_str : str
            The string representing the URDF.
        root_joint : pinocchio.JointModel or None
            The root joint of the model (optional).
        tool_name : str or None
            The name of the robot's tool.
        gravity : np.ndarray, shape (3,) or None
            The gravity vector. If ``None``, defaults to pinocchio's ``[0, 0,
            -9.81]``.
        """
        if root_joint is not None:
            model = pinocchio.buildModelFromXML(urdf_str, root_joint)
        else:
            model = pinocchio.buildModelFromXML(urdf_str)
        geom_model = pinocchio.buildGeomFromUrdfString(
            model, urdf_str, pinocchio.COLLISION
        )
        return cls(model, geom_model, tool_link_name=tool_link_name, gravity=gravity)

    @classmethod
    def from_urdf_file(
        cls, urdf_file_path, root_joint=None, tool_link_name=None, gravity=None
    ):
        """Load the model directly from a URDF file."""
        with open(urdf_file_path) as f:
            urdf_str = f.read()
        return cls.from_urdf_string(
            urdf_str, root_joint, tool_link_name=tool_link_name, gravity=gravity
        )

    def get_frame_index(self, name):
        """Get the index of a frame by name.

        Parameters
        ----------
        name : str
            The name of the frame.

        Returns
        -------
        : int
            The index of the frame.

        Raises
        ------
        ValueError
            If the frame does not exist.
        """
        if not self.model.existFrame(name):
            raise ValueError(f"Model has no frame named {name}.")
        return self.model.getFrameId(name)

    def _resolve_frame_index(self, frame):
        if frame is None:
            return self.tool_idx
        if isinstance(frame, str):
            return self.get_frame_index(frame)
        return frame

    def compute_forward_kinematics(self, q, v=None, a=None):
        """Compute forward kinematics.

        This must be called before any calls to obtain task-space quantities,
        such as ``get_frame_pose``, ``get_frame_velocity``, etc.

        Parameters
        ----------
        q : np.ndarray, shape (self.nq,)
            The joint positions.
        v : np.ndarray, shape (self.nv,)
            The joint velocities.
        a : np.ndarray, shape (self.nv,)
            The joint accelerations.
        """
        if v is None:
            v = np.zeros(self.nv)
        if a is None:
            a = np.zeros(self.nv)

        assert q.shape == (self.nq,)
        assert v.shape == (self.nv,)
        assert a.shape == (self.nv,)

        pinocchio.forwardKinematics(self.model, self.data, q, v, a)
        pinocchio.updateFramePlacements(self.model, self.data)

    def compute_joint_torques(self, q, v, a):
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
        assert q.shape == (self.nq,)
        assert v.shape == (self.nv,)
        assert a.shape == (self.nv,)

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

    def get_frame_pose(self, frame=None):
        """Get the pose of a frame.

        Note that ``compute_forward_kinematics(q, ...)`` must be called first.

        Parameters
        ----------
        frame : int or str or None
            If ``int``, this is interpreted as the frame index. If ``str``,
            interpreted as the name of a frame. If ``None``, defaults to
            ``self.tool_idx``.

        Returns
        -------
        : tuple
            Returns a tuple (position, orientation). ``position`` is a
            np.ndarray of shape (3,) and ``orientation`` is a rotation matrix
            represented by an np.ndarray of shape (3, 3).
        """
        idx = self._resolve_frame_index(frame)
        pose = self.data.oMf[idx]
        return pose.rotation, pose.translation

    def get_frame_velocity(self, link_idx=None, frame="local_world_aligned"):
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

    def get_frame_classical_acceleration(self, link_idx=None, frame="local_world_aligned"):
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

    def get_frame_spatial_acceleration(self, link_idx=None, frame="local_world_aligned"):
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

    # TODO we have a conflict
    def compute_frame_jacobian(self, q, frame=None, expressed_in="local_world_aligned"):
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
