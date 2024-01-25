import numpy as np
import sympy

import inertial_params as ip

import IPython

# a, b, c = sympy.symbols("a,b,c")
# ms = sympy.symbols("m1,m2,m3,m4,m5,m6,m7,m8")
#
# vs = [
#     sympy.Matrix([a, b, c]),
#     sympy.Matrix([a, b, -c]),
#     sympy.Matrix([a, -b, c]),
#     sympy.Matrix([a, -b, -c]),
#     sympy.Matrix([-a, b, c]),
#     sympy.Matrix([-a, b, -c]),
#     sympy.Matrix([-a, -b, c]),
#     sympy.Matrix([-a, -b, -c]),
# ]
#
# H = sympy.Matrix.zeros(3, 3)
# h = sympy.Matrix.zeros(3, 1)
# for m, v in zip(ms, vs):
#     h += m * v
#     H += m * v * v.transpose()

box = ip.AxisAlignedBox(half_extents=[0.5, 0.5, 0.5])
vertices = box.vertices

m = 1.0
c = np.array([0, 0, 0])
h = m * c

mv = m * vertices / 8
A = np.vstack((np.ones((1, 8)), vertices.T))
b = np.concatenate(([m], h))

μ = np.linalg.lstsq(A, b, rcond=None)[0]

# A = np.array([[1, 1], [-0.5, 0.5]])
# bx = np.array([m, h[0]])
# mx = np.linalg.solve(A, bx)
#
# by = np.array([m, h[1]])
# my = np.linalg.solve(A, by)
#
# bz = np.array([m, h[2]])
# mz = np.linalg.solve(A, bz)

P = ip.InertialParameters.from_point_masses(masses=μ, points=vertices)
P0 = ip.InertialParameters.from_point_masses(masses=np.ones(8) / 8, points=vertices)
# vs = np.array([[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5]])
# P1 = ip.InertialParameters.from_point_masses(masses=np.ones(6) / 6, points=vs)

ps = np.array([[0.5, 0.5, 1.0]])
n = len(ps)
P1 = ip.InertialParameters.from_point_masses(masses=np.ones(n) / n, points=ps)

IPython.embed()
