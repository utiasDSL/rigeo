"""Verify the linear program conditions for cuboid density realizability
verification for two- and three-dimensional problems."""
import sympy as sym

import IPython


def twodim():
    """Two-dimensional version of the problem."""
    # half extents of the cuboid
    rx, ry = sym.symbols("rx,ry")
    rxy = rx * ry

    Hxy, hx, hy, m = sym.symbols("Hxy,hx,hy,m")
    m = 1
    b = sym.Matrix([Hxy, hx, hy, m])
    z = sym.Matrix(sym.symbols("z1,z2,z3,z4"))

    J = sym.Matrix([[m, Hxy, hx], [Hxy, m, hy], [hx, hy, m]])

    A = sym.Matrix(
        [[rxy, -rxy, -rxy, rxy], [x, -x, x, -x], [y, y, -y, -y], [1, 1, 1, 1]]
    )

    sol, params = A.gauss_jordan_solve(b)

    IPython.embed()


def threedim():
    """Three-dimensional version of the problem."""

    # half extents of the cuboid
    rx, ry, rz = sym.symbols("rx,ry,rz")

    # inertial parameters
    Hxy, Hxz, Hyz, hx, hy, hz, m = sym.symbols("Hxy,Hxz,Hyz,hx,hy,hz,m")

    # fmt: off
    vertices = sym.Matrix([
        [ rx,  ry,  rz, 1],
        [ rx,  ry, -rz, 1],
        [ rx, -ry,  rz, 1],
        [ rx, -ry, -rz, 1],
        [-rx,  ry,  rz, 1],
        [-rx,  ry, -rz, 1],
        [-rx, -ry,  rz, 1],
        [-rx, -ry, -rz, 1],
    ])
    # fmt: on

    # setup the system to be solved
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

    # build the solution to Aμ = b
    x, y, z = sym.symbols("x,y,z")
    ρ = sym.symbols("ρ")
    φ = sym.Matrix([
        (rx + x) * (ry + y) * (rz + z),
        (rx + x) * (ry + y) * (rz - z),
        (rx + x) * (ry - y) * (rz + z),
        (rx + x) * (ry - y) * (rz - z),
        (rx - x) * (ry + y) * (rz + z),
        (rx - x) * (ry + y) * (rz - z),
        (rx - x) * (ry - y) * (rz + z),
        (rx - x) * (ry - y) * (rz - z),
    ]) / (8 * rx * ry * rz)

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

    IPython.embed()


threedim()
