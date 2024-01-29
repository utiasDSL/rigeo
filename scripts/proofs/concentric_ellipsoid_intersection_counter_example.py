"""Counter-example for general concentric ellipsoid intersection."""
import numpy as np
from spatialmath.base import roty

import inertial_params as ip

import IPython

C = roty(-np.pi / 4)
box = ip.Box(half_extents=[1, 0.5, 0.5], center=[0, 0, 0])
ell1 = ip.maximum_inscribed_ellipsoid(box.vertices)
ell2 = ell1.transform(C=C)

# points are not contained in the intersection, so params are not realizable on
# the intersection. However, they are realizable on each of the individual
# ellipsoids.
p0 = np.array([0.75, 0, 0])
p1 = C @ p0
P0 = ip.InertialParameters.from_point_masses(masses=[0.5, 0.5], points=[p0, p1])
