"""Symbolic derivation of box edge inertia."""
import sympy

import IPython


def box_edge_inertia():
    """Derive box edge inertia using parallel axis theorem."""
    # mass of each edge is proportional to its length
    a, b, c = sympy.symbols("a, b, c")
    L = a + b + c

    mx = a / L
    Ex = sympy.Matrix.diag([a**2, 0, 0])
    rxs = [
        sympy.Matrix([0, b, c]),
        sympy.Matrix([0, -b, c]),
        sympy.Matrix([0, b, -c]),
        sympy.Matrix([0, -b, -c]),
    ]
    Hx = mx * (Ex / 3 + sympy.Add(*[rx * rx.transpose() / 4 for rx in rxs]))

    my = b / L
    Ey = sympy.Matrix.diag([0, b**2, 0])
    rys = [
        sympy.Matrix([a, 0, c]),
        sympy.Matrix([-a, 0, c]),
        sympy.Matrix([a, 0, -c]),
        sympy.Matrix([-a, 0, -c]),
    ]
    Hy = my * (Ey / 3 + sympy.Add(*[ry * ry.transpose() / 4 for ry in rys]))

    mz = c / L
    Ez = sympy.Matrix.diag([0, 0, c**2])
    rzs = [
        sympy.Matrix([a, b, 0]),
        sympy.Matrix([-a, b, 0]),
        sympy.Matrix([a, -b, 0]),
        sympy.Matrix([-a, -b, 0]),
    ]
    Hz = mz * (Ez / 3 + sympy.Add(*[rz * rz.transpose() / 4 for rz in rzs]))

    H = Hx + Hy + Hz
    I = sympy.trace(H) * sympy.eye(3) - H
    print(f"H =\n{H}")
    print(f"I =\n{sympy.simplify(I)}")

    # when all half extents equal, this is a cube
    r = sympy.symbols("r")
    H_cube = H.subs({a: r, b: r, c: r})
    I_cube = sympy.trace(H_cube) * sympy.eye(3) - H_cube

    E = r**2 * sympy.eye(3)
    S = sympy.trace(E) * sympy.eye(3) - E
    assert H_cube == 7 * E / 9
    assert I_cube == 7 * S / 9


box_edge_inertia()
