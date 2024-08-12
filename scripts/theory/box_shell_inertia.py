"""Symbolic derivation of box shell inertia."""
import sympy


def box_shell_inertia():
    δ = sympy.symbols("δ")
    a, b, c = sympy.symbols("a, b, c")
    S = sympy.Matrix.diag(b**2 + c**2, a**2 + c**2, a**2 + b**2)

    a2 = a + δ
    b2 = b + δ
    c2 = c + δ
    S2 = sympy.Matrix.diag(
        b2**2 + c2**2, a2**2 + c2**2, a2**2 + b2**2
    )

    nom = a2 * b2 * c2 * S2 - a * b * c * S
    den = a2 * b2 * c2 - a * b * c
    I = (nom / den).limit(δ, 0) / 3

    # compute the second moment matrix too
    H = sympy.trace(I) * sympy.eye(3) / 2 - I
    H.simplify()
    return H, I


def box_shell_inertia_parax():
    """Derive box shell inertia using parallel axis theorem."""
    # mass of each face is proportional to its area
    a, b, c = sympy.symbols("a, b, c")
    A1 = 2 * 4 * b * c
    A2 = 2 * 4 * a * c
    A3 = 2 * 4 * a * b
    A = A1 + A2 + A3
    m1 = A1 / A
    m2 = A2 / A
    m3 = A3 / A

    # use parallel axis theorem for the second moment matrix
    E1 = sympy.Matrix.diag([0, b**2, c**2])
    H1 = m1 * E1 / 3 + m1 * sympy.Matrix.diag([a**2, 0, 0])

    E2 = sympy.Matrix.diag([a**2, 0, c**2])
    H2 = m2 * E2 / 3 + m2 * sympy.Matrix.diag([0, b**2, 0])

    E3 = sympy.Matrix.diag([a**2, b**2, 0])
    H3 = m3 * E3 / 3 + m3 * sympy.Matrix.diag([0, 0, c**2])

    H = H1 + H2 + H3
    I = sympy.trace(H) * sympy.eye(3) - H
    return H, I


# check that the two derivations are equal
H1, I1 = box_shell_inertia()
H2, I2 = box_shell_inertia_parax()
diff = I2 - I1
diff.simplify()
assert diff == sympy.zeros(3, 3)

print(f"H =\n{H1}")
print(f"I =\n{I1}")

# when all half extents equal, this is a cube
a, b, c = sympy.symbols("a, b, c")
r = sympy.symbols("r")
H_cube = H1.subs({a: r, b: r, c: r})
I_cube = sympy.trace(H_cube) * sympy.eye(3) - H_cube

E = r**2 * sympy.eye(3)
S = sympy.trace(E) * sympy.eye(3) - E
assert H_cube == 5 * E / 9
assert I_cube == 5 * S / 9
