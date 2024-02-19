"""Compute the minimum distance and closest points between two convex shapes."""
from dataclasses import dataclass

import numpy as np
import cvxpy as cp


@dataclass
class ClosestPointInfo:
    """Information about a closest point query.

    Parameters
    ----------
    p1 : np.ndarray
        The closest point on the first shape.
    p2 : np.ndarray
        The closest point on the second shape.
    dist : float, non-negative
        The distance between the two shapes.
    """

    p1: np.ndarray
    p2: np.ndarray
    dist: float


def closest_points(shape1, shape2, solver=None):
    """Compute the closest points between two shapes.

    When the two shapes are in contact or penetrating, the distance will be
    zero and the points can be anything inside the intersection.

    This function is *not* optimized for speed: a full convex program is
    solved. Useful for prototyping but not for high-speed queries; in the
    latter case, use something like hpp-fcl.

    Parameters
    ----------
    shape1 : ConvexPolyhedron or Ellipsoid or Cylinder
        The first shape to check.
    shape2 : ConvexPolyhedron or Ellipsoid or Cylinder
        The second shape to check.
    solver : str or None
        The solver for cvxpy to use.

    Returns
    -------
    : ClosestPointInfo
        Information about the closest points.
    """
    p1 = cp.Variable(3)
    p2 = cp.Variable(3)

    objective = cp.Minimize(cp.norm2(p2 - p1))
    constraints = shape1.must_contain(p1) + shape2.must_contain(p2)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    return ClosestPointInfo(p1=p1.value, p2=p2.value, dist=objective.value)
