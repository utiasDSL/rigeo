"""Verify the proof of the necessary conditions for density realizability on a
cuboid."""
import sympy as sym


# half extents of the cuboid
rx, ry, rz = sym.symbols("rx,ry,rz")

# inertial parameters
Hxy, Hxz, Hyz, hx, hy, hz, m = sym.symbols("Hxy,Hxz,Hyz,hx,hy,hz,m")

# vertices of the cuboid
vertices = sym.Matrix(
    [
        [rx, ry, rz, 1],
        [rx, ry, -rz, 1],
        [rx, -ry, rz, 1],
        [rx, -ry, -rz, 1],
        [-rx, ry, rz, 1],
        [-rx, ry, -rz, 1],
        [-rx, -ry, rz, 1],
        [-rx, -ry, -rz, 1],
    ]
)

# setup the linear system to be solved
A = sym.Matrix.zeros(7, 8)
for i in range(A.cols):
    V = vertices[i, :].T * vertices[i, :]
    A[0, i] = V[0, 1]
    A[1, i] = V[0, 2]
    A[2, i] = V[1, 2]
    A[3, i] = V[0, 3]
    A[4, i] = V[1, 3]
    A[5, i] = V[2, 3]
    A[6, i] = V[3, 3]

b = sym.Matrix([Hxy, Hxz, Hyz, hx, hy, hz, m])

# build the proposed solution μ for Aμ = b
x, y, z = sym.symbols("x,y,z")
ρ = sym.symbols("ρ")
φ = sym.Matrix(
    [
        (rx + x) * (ry + y) * (rz + z),
        (rx + x) * (ry + y) * (rz - z),
        (rx + x) * (ry - y) * (rz + z),
        (rx + x) * (ry - y) * (rz - z),
        (rx - x) * (ry + y) * (rz + z),
        (rx - x) * (ry + y) * (rz - z),
        (rx - x) * (ry - y) * (rz + z),
        (rx - x) * (ry - y) * (rz - z),
    ]
) / (8 * rx * ry * rz)

α = sym.symbols("α")  # placeholder that will cancel out
ρφ = (ρ * φ).expand()

# "integrate" by substituting inertial parameters for some expressions
# do more complex expressions first to avoid clobbering
ρφ = ρφ.subs({ρ * x * y * z: α})
ρφ = ρφ.subs({ρ * x * y: Hxy, ρ * x * z: Hxz, ρ * y * z: Hyz})
ρφ = ρφ.subs({ρ * x: hx, ρ * y: hy, ρ * z: hz})
μ = ρφ.subs({ρ: m})

# verify the solution
Aμ = A * μ
Aμ.simplify()
assert Aμ == b
