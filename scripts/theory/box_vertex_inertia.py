"""Symbolic derivation of box inertia when mass evenly distributed at vertices."""
import sympy


def box_vertex_inertia():
    a, b, c = sympy.symbols("a, b, c")
    vertices = [
        [a, b, c],
        [a, b, -c],
        [a, -b, c],
        [a, -b, -c],
        [-a, b, c],
        [-a, b, -c],
        [-a, -b, c],
        [-a, -b, -c],
    ]
    vertices = [sympy.Matrix(v) for v in vertices]

    H = sympy.Add(*[v * v.transpose() / 8 for v in vertices])
    I = sympy.trace(H) * sympy.eye(3) - H
    print(f"H =\n{H}")
    print(f"I =\n{I}")

    # when all half extents equal, this is a cube
    r = sympy.symbols("r")
    H_cube = H.subs({a: r, b: r, c: r})
    I_cube = sympy.trace(H_cube) * sympy.eye(3) - H_cube

    E = r**2 * sympy.eye(3)
    S = sympy.trace(E) * sympy.eye(3) - E
    assert H_cube == E
    assert I_cube == S


box_vertex_inertia()
