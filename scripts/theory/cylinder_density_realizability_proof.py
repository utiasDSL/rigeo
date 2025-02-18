"""Verify the proof of the necessary conditions for density realizability on a
cuboid."""
import sympy as sym
import IPython


# half extents of the cuboid
rx, ry, rz = sym.symbols("rx,ry,rz")
px, py, pz = sym.symbols("px,py,pz")
α = sym.symbols("α")

q1 = sym.Matrix([px, py, rz, 1])
Q1 = q1 * q1.T

q2 = sym.Matrix([px, py, -rz, 1])
Q2 = q2 * q2.T

P = ((rz + pz) * Q1 + (rz - pz) * Q2) / (2 * rz)
P.simplify()

IPython.embed()
