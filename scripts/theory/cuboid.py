"""Examining the cuboid special case."""
import numpy as np
import rigeo as rg
import IPython

mass = 1
box = rg.Box(half_extents=[0.5, 1, 2])

points1 = box.vertices
masses1 = mass * np.ones(8) / 8
params1 = rg.InertialParameters.from_point_masses(masses=masses1, points=points1)

# this has same mass and CoM but different H
# (but the diagonal elements are the same)
masses2 = mass * np.ones(4) / 4
points2 = np.array([[0.5, 1, 2], [0.5, -1, -2], [-0.5, 1, 2], [-0.5, -1, -2]])
params2 = rg.InertialParameters.from_point_masses(masses=masses2, points=points2)

IPython.embed()
