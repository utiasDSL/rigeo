from pathlib import Path

import numpy as np
import hppfcl
import pinocchio

from rigeo.shape import Box, Ellipsoid, Cylinder
from rigeo.inertial import I2H, InertialParameters
from rigeo.rigidbody import RigidBody


# TODO want something more elegant than this
# def _ref_frame_from_string(s):
#     """Translate a string to pinocchio's ReferenceFrame enum value."""
#     s = s.lower()
#     if s == "local":
#         return pinocchio.ReferenceFrame.LOCAL
#     if s == "local_world_aligned":
#         return pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
#     if s == "world":
#         return pinocchio.ReferenceFrame.WORLD
#     raise ValueError(f"{s} is not a valid Pinocchio reference frame.")


RF = pinocchio.ReferenceFrame


def _hppfcl_to_shape(geom):
    """Convert an HPP-FCL shape to the equivalent rigeo shape."""
    type_ = geom.getNodeType()

    if type_ == hppfcl.hppfcl.NODE_TYPE.GEOM_CONVEX:
        raise NotImplementedError("Mesh is not yet supported.")
        pass
    elif type_ == hppfcl.hppfcl.NODE_TYPE.GEOM_BOX:
        return Box(half_extents=geom.halfSide)
    elif type_ == hppfcl.hppfcl.NODE_TYPE.GEOM_SPHERE:
        return Ellipsoid.sphere(radius=geom.radius)
    elif type_ == hppfcl.hppfcl.NODE_TYPE.GEOM_CYLINDER:
        return Cylinder(length=2 * geom.halfLength, radius=geom.radius)
    else:
        raise ValueError("Unrecognized shape.")


def _pin_to_shape(geom_obj):
    shape = _hppfcl_to_shape(geom_obj.geometry)
    return shape.transform(
        rotation=geom_obj.placement.rotation, translation=geom_obj.placement.translation
    )


class MultiBody:
    """A connected set of rigid bodies."""

    def __init__(self, model, geom_model, tool_link_name=None, gravity=None):
        self.model = model
        self.data = self.model.createData()

        self.nj = model.njoints  # number of joints
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

        # bodies is a mapping of joint indices to the rigid bodies composing the
        # multibody
        self.bodies = {}
        for i in range(geom_model.ngeoms):
            geom = geom_model.geometryObjects[i]
            joint_idx = geom.parentJoint
            inertia = model.inertias[joint_idx]

            shape = _pin_to_shape(geom)

            if joint_idx in self.bodies:
                self.bodies[joint_idx].shapes.append(shape)
            else:
                mass = inertia.mass
                h = mass * inertia.lever
                Hc = I2H(inertia.inertia)
                params = InertialParameters.translate_from_com(mass, h, Hc)
                self.bodies[joint_idx] = RigidBody(shapes=[shape], params=params)

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

    def get_joint_index(self, name):
        # TODO
        if not self.model.existJointName(name):
            raise ValueError(f"Model has no joint named {name}.")
        return self.model.getJointId(name)

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

    def _resolve_joint_index(self, joint):
        """Convert joint index or name to index."""
        if isinstance(joint, str):
            return self.get_joint_index(joint)
        return joint

    def _resolve_frame_index(self, frame):
        """Convert frame index or name to index."""
        if frame is None:
            return self.tool_idx
        if isinstance(frame, str):
            return self.get_frame_index(frame)
        return frame

    def get_bodies(self, joints):
        """Get the rigid bodies corresponding to the given joints.

        Parameters
        ----------
        joints : list[str] or list[int]
            Joints to get the bodies for. Can either be specified by name or
            index.

        Returns
        -------
        : list[RigidBody]
            A list of rigid bodies corresponding to the joints.
        """
        indices = [self._resolve_joint_index(joint) for joint in joints]
        return [self.bodies[idx] for idx in indices]

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
        """Compute the joint torque regressor matrix for the multibody.

        The joint torque regressor matrix maps the stacked vector of link
        inertial parameters to the joint torques.

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
        : np.ndarray, shape (self.nv, 10 * self.nlinks)
            The joint torque regressor matrix.
        """
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

    def get_frame_velocity(self, frame=None, expressed_in=RF.LOCAL):
        """Get velocity of link at index link_idx"""
        idx = self._resolve_frame_index(frame)
        V = pinocchio.getFrameVelocity(
            self.model,
            self.data,
            idx,
            expressed_in,
        )
        return V.linear, V.angular

    def get_frame_classical_acceleration(self, frame=None, expressed_in=RF.LOCAL):
        """Get the classical acceleration of a link."""
        idx = self._resolve_frame_index(frame)
        A = pinocchio.getFrameClassicalAcceleration(
            self.model,
            self.data,
            idx,
            expressed_in,
        )
        return A.linear, A.angular

    def get_frame_spatial_acceleration(self, frame=None, expressed_in=RF.LOCAL):
        """Get the spatial acceleration of a link."""
        idx = self._resolve_frame_index(frame)
        A = pinocchio.getFrameAcceleration(
            self.model,
            self.data,
            idx,
            expressed_in,
        )
        return A.linear, A.angular

    def compute_frame_jacobian(self, q, frame=None, expressed_in=RF.LOCAL):
        """Compute the robot geometric Jacobian."""
        idx = self._resolve_frame_index(frame)
        return pinocchio.computeFrameJacobian(
            self.model,
            self.data,
            q,
            idx,
            expressed_in,
        )
