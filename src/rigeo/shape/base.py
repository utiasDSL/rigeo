"""Abstract base class for geometric shapes."""
import abc


class Shape(abc.ABC):
    @abc.abstractmethod
    def contains(self, points, tol=1e-8):
        """Test if the shape contains a set of points.

        Parameters
        ----------
        points : np.ndarray, shape (n, self.dim)
            The points to check.
        tol : float, non-negative
            The numerical tolerance for membership.

        Returns
        -------
        : bool or np.ndarray of bool, shape (n,)
            Boolean array where each entry is ``True`` if the shape
            contains the corresponding point and ``False`` otherwise.
        """
        pass

    def contains_polyhedron(self, poly, tol=1e-8):
        """Check if this shape contains a polyhedron.

        Parameters
        ----------
        poly : ConvexPolyhedron
            The polyhedron to check.
        tol : float, non-negative
            The numerical tolerance for membership.

        Returns
        -------
        : bool
            ``True`` if this shapes contains the polyhedron, ``False`` otherwise.
        """
        return self.contains(poly.vertices, tol=tol).all()

    @abc.abstractmethod
    def must_contain(self, points, scale=1.0):
        """Generate cvxpy constraints to keep the points inside the shape.

        Parameters
        ----------
        points : cp.Variable, shape (self.dim,) or (n, self.dim)
            A point or set of points to constrain to lie inside the shape.
        scale : float, positive
            Scale for ``points``. The main idea is that one may wish to check
            that the CoM belongs to the shape, but using the quantity
            :math:`h=mc`. Then ``must_contain(c)`` is equivalent to
            ``must_contain(h, scale=m)``.

        Returns
        -------
        : list
            A list of cxvpy constraints that keep the points inside the shape.
        """
        pass

    # @abc.abstractmethod
    # def can_realize(self, params, eps=0, **kwargs):
    #     """Check if the shape can realize the inertial parameters.
    #
    #     Parameters
    #     ----------
    #     params : InertialParameters
    #         The inertial parameters to check.
    #     eps : float
    #         The parameters will be considered consistent if all of the
    #         eigenvalues of the pseudo-inertia matrix are greater than or equal
    #         to ``eps``.
    #
    #     Additional keyword arguments are passed to the solver, if one is needed.
    #
    #     Returns
    #     -------
    #     : bool
    #         ``True`` if the parameters are realizable, ``False`` otherwise.
    #     """
    #     pass
    #
    # @abc.abstractmethod
    # def must_realize(self, param_var, eps=0):
    #     """Generate cvxpy constraints for inertial parameters to be realizable
    #     on this shape.
    #
    #     Parameters
    #     ----------
    #     param_var : cp.Expression, shape (4, 4) or shape (10,)
    #         The cvxpy inertial parameter variable. If shape is ``(4, 4)``, this
    #         is interpreted as the pseudo-inertia matrix. If shape is ``(10,)``,
    #         this is interpreted as the inertial parameter vector.
    #     eps : float
    #         Pseudo-inertia matrix ``J`` is constrained such that ``J - eps *
    #         np.eye(4)`` is positive semidefinite and J is symmetric.
    #
    #     Returns
    #     -------
    #     : list
    #         List of cvxpy constraints.
    #     """
    #     pass

    @abc.abstractmethod
    def aabb(self):
        """Generate the minimum-volume axis-aligned box that bounds the shape.

        Returns
        -------
        : Box
            The axis-aligned bounding box.
        """
        pass

    @abc.abstractmethod
    def mbe(self, rcond=None, sphere=False, solver=None):
        """Generate the minimum-volume bounding ellipsoid for the shape.

        Parameters
        ----------
        sphere : bool
            If ``True``, force the ellipsoid to be a sphere.
        solver : str or None
            If generating the minimum bounding ellipsoid requires solving an
            optimization problem, a solver can optionally be specified.

        Returns
        -------
        : Ellipsoid
            The minimum bounding ellipsoid (or sphere, if ``sphere=True``).
        """
        pass

    @abc.abstractmethod
    def random_points(self, shape=1, rng=None):
        """Generate random points contained in the shape.

        Parameters
        ----------
        shape : int or tuple
            The shape of the set of points to be returned.

        Returns
        -------
        : np.ndarray, shape ``shape + (self.dim,)``
            The random points.
        """
        pass

    def grid(self, n):
        """Generate a regular grid inside the shape.

        The approach is to compute a bounding box, generate a grid for that,
        and then discard any points not inside the actual polyhedron.

        Parameters
        ----------
        n : int
            The maximum number of points along each dimension.

        Returns
        -------
        : np.ndarray, shape (N, self.dim)
            The points contained in the grid.
        """
        assert n > 0
        box_grid = self.aabb().grid(n)
        contained = self.contains(box_grid)
        return box_grid[contained, :]

    @abc.abstractmethod
    def transform(self, rotation=None, translation=None):
        """Apply a rigid transform to the shape.

        Parameters
        ----------
        rotation : np.ndarray, shape (d, d)
            Rotation matrix.
        translation : np.ndarray, shape (d,)
            Translation vector.

        Returns
        -------
        : Shape
            A new shape that has been rigidly transformed.
        """
        pass

    @abc.abstractmethod
    def is_same(self, other, tol=1e-8):
        """Check if this shape is the same as another one.

        Parameters
        ----------
        other : Shape
            The other shape to check.
        tol : float, non-negative
            The numerical tolerance for membership.

        Returns
        -------
        : bool
            ``True`` if the polyhedra are the same, ``False`` otherwise.
        """
        pass
