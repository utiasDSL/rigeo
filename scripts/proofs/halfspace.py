"""This script tests the set of valid inertial parameters on a halfspace.

It appears that the only conditions required are m > 0 and c is in the
halfspace; H can be as large as desired by putting the point mass at infinity.
"""
import numpy as np

# halfspace is {p | x @ p >= 0}
x = np.array([1, 0, 0])
m = 1
c = 1
h = m * c

# reducing m2 leads to increasing v2
# we see that H blows up
m2 = 1e-9
m1 = m - m2
v1 = 0
v2 = (h + (m2 - m) * v1) / m2
H = m1 * v1**2 + m2 * v2**2

assert np.isclose(h, m1 * v1 + m2 * v2)
print(f"v2 = {v2}")
print(f"H  = {H}")
