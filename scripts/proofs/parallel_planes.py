"""Parallel planes example using the ellipsoid formulation."""
import numpy as np

import inertial_params as ip

import IPython


v1 = np.zeros(3)
v2 = np.array([1, 0, 0])
normal = np.array([1, 0, 0])
a = (v1 + v2) / 2  # midpoint
r = np.linalg.norm(a - v1)  # radius

# construct (degenerate) ellipsoid
A = np.outer(normal, normal) / r**2
ell = ip.Ellipsoid(Einv=A, c=a)

# construct params right on the edge of the space
params = ip.InertialParameters(mass=1.0, h=v1, H=0 * np.eye(3))

print(f"tr(QJ) = {np.trace(ell.Q @ params.J)}")
