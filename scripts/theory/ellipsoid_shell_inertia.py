"""Symbolic derivation of ellipsoid shell inertia."""
import sympy


def sphere_shell_inertia():
    r, δ = sympy.symbols("r, δ")
    expr = ((r + δ) ** 5 - r**5) / ((r + δ) ** 3 - r**3)
    print(expr.limit(δ, 0))


def ellipsoid_shell_inertia():
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
    I = (nom / den).limit(δ, 0) / 5
    print(f"I = {I}")

    # compute the second moment matrix too
    H = sympy.trace(I) * sympy.eye(3) / 2 - I
    H.simplify()
    print(f"H = {H}")

    # confirm that this is the same as the sphere when all semi-axes are equal
    r = sympy.symbols("r")
    print(I.subs({a: r, b: r, c: r}))


sphere_shell_inertia()
ellipsoid_shell_inertia()
