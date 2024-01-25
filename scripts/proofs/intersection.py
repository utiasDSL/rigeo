import numpy as np
import sympy
from spatialmath.base import roty

import inertial_params as ip

import IPython

box1 = ip.AxisAlignedBox(half_extents=[0.5, 0.5, 0.5], center=[-0.25, 0, 0])
box2 = ip.AxisAlignedBox(half_extents=[0.5, 0.5, 0.5], center=[0.25, 0, 0])

# m = 1.0
# c = np.array([0, 0, 0])
# h = m * c

p0 = np.array([[0, 0.5, 0], [0.5, 0, 0]])
P0 = ip.InertialParameters.from_point_masses(masses=[0.5, 0.5], points=p0)

p1 = np.array([[-0.5, 1, 0], [0.5, 0, 0]])
P1 = ip.InertialParameters.from_point_masses(masses=[0.25, 0.75], points=p1)

###

r = 0.5
x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.array([0, 0, 1])

box = ip.AxisAlignedBox(half_extents=[r, r, r])

Ax = np.outer(x, x) / r**2
Ay = np.outer(y, y) / r**2
Az = np.outer(z, z) / r**2

a = np.zeros(3)

ellx = ip.Ellipsoid(Einv=Ax, c=a)
elly = ip.Ellipsoid(Einv=Ay, c=a)
ellz = ip.Ellipsoid(Einv=Az, c=a)

# m = 1.0
# c = np.array([0, 0, 0])
# h = m * c

# p0 = np.array([[0, 0.5, 0], [0.5, 0, 0]])
P0 = ip.InertialParameters.from_point_masses(masses=np.ones(8) / 8, points=box.vertices)
#
# p1 = np.array([[-0.5, 1, 0], [0.5, 0, 0]])
# P1 = ip.InertialParameters.from_point_masses(masses=[0.25, 0.75], points=p1)

Hs = ip.I2H(ip.solid_sphere_inertia_matrix(1.0, 1.0))

ps = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
P1 = ip.InertialParameters.from_point_masses(masses=np.ones(6) / 6, points=ps)

IPython.embed()

###

# this is a counter-example for general concentric ellipsoid intersection
C = roty(-np.pi / 4)
box = ip.AxisAlignedBox(half_extents=[1, 0.5, 0.5], center=[0, 0, 0])
ell1 = ip.maximum_inscribed_ellipsoid(box.vertices)
ell2 = ell1.transform(C=C)

# points are not contained in the intersection, so params are not realizable on
# the intersection. However, they are realizable on each of the individual
# ellipsoids.
p0 = np.array([0.75, 0, 0])
p1 = C @ p0
P0 = ip.InertialParameters.from_point_masses(masses=[0.5, 0.5], points=[p0, p1])
