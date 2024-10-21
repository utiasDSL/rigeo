import itertools
import math

import numpy as np
import cvxpy as cp
import rigeo as rg


class MomentIndex:
    """MomentIndex for a term in a moment sequence or polynomial."""

    def __init__(self, value):
        self.value = tuple(int(x) for x in value)

    def __str__(self):
        return "".join(str(x) for x in self.value)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, MomentIndex):
            return False
        return self.value == other.value

    def __add__(self, other):
        if type(other) is int and other == 0:
            return self
        assert len(other.value) == len(self.value)
        value = (a + b for a, b in zip(self.value, other.value))
        return MomentIndex(value)

    def __radd__(self, other):
        # addition is commutative
        return self + other

    def __mul__(self, other):
        return self + other


class Polynomial:
    """Polynomial function."""

    def __init__(self, coefficients, tol=0):
        self.coefficients = {}
        for key, coeff in coefficients.items():
            if not isinstance(key, MomentIndex):
                key = MomentIndex(key)
            # get rid of coefficients near zero
            if not np.isclose(coeff, 0, rtol=0, atol=tol):
                self.coefficients[key] = coeff

    @classmethod
    def zeros(cls, n, d):
        """Create a polynomial of degree ``d`` in ``n`` dimensions with all coefficients zero."""
        b = cls.basis(n, d)
        return cls({idx: 0 for idx in b})

    @classmethod
    def ones(cls, n, d):
        """Create a polynomial of degree ``d`` in ``n`` dimensions with all coefficients ones."""
        b = cls.basis(n, d)
        return cls({idx: 1 for idx in b})

    @classmethod
    def affine(cls, a, b, tol=0):
        """Create a polynomial g(x) >= 0 from affine constraint a @ x <= b"""
        n = len(a)
        coefficients = {(0,) * n: b}
        for i in range(n):
            idx = np.zeros(n, dtype=int)
            idx[i] = 1
            coefficients[tuple(idx)] = -a[i]
        return cls(coefficients, tol=tol)

    @property
    def degree(self):
        """The degree of the polynomial."""
        return np.max([np.sum(idx.value) for idx in self.coefficients.keys()])

    def __str__(self):
        strs = [f"{c} * x^{idx}" for idx, c in self.coefficients.items()]
        return " + ".join(strs)

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        coefficients = self.coefficients.copy()
        for idx, coeff in other.coefficients.items():
            if idx in coefficients:
                coefficients[idx] += coeff
            else:
                coefficients[idx] = coeff
        return Polynomial(coefficients)

    def localize(self, idx):
        """Localize this polynomial about the given moment index term."""
        return Polynomial(
            {s + idx: coeff for s, coeff in self.coefficients.items()}
        )

    def substitute(self, sub_var_map):
        """Substitute variables or actual values into the polynomial."""
        return cp.sum(
            [
                coeff * sub_var_map[idx]
                for idx, coeff in self.coefficients.items()
            ]
        )

    def evaluate(self, x):
        """Evaluate the polynomial at point x."""
        x = np.array(x, copy=False)
        return np.sum(
            [
                coeff * np.prod(x**idx.value)
                for idx, coeff in self.coefficients.items()
            ]
        )

    @staticmethod
    def basis(n, d):
        """Construct the basis vector for a polynomial of degree at most d in n dimensions.

        See equation (3.9) of Lasserre's book.
        """
        basis = [MomentIndex(np.zeros(n, dtype=int))]
        for i in range(n):
            b = np.zeros(n, dtype=int)
            b[i] = 1
            basis.append(MomentIndex(b))
        return [
            sum(group)
            for group in itertools.combinations_with_replacement(basis, d)
        ]

    @staticmethod
    def dim(n, d):
        """Dimension of the polynomial basis vector."""
        return math.comb(n + d, d)


class MomentMatrix:
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.r = Polynomial.dim(n, d)
        self.shape = (self.r, self.r)

        # symbolic form of the moment matrix
        b = Polynomial.basis(n, d)
        self.indices = np.outer(b, b)

    def __str__(self):
        return str(self.indices)

    def __repr__(self):
        return str(self)

    def _compute_var_map(self, M_var):
        """Compute the mapping from moment indices to values in the provided matrix.

        The provided matrix may be either a cvxpy variable or a numpy array.
        """
        assert M_var.shape == self.shape
        sub_var_map = {}
        for i in range(self.r):
            for j in range(i, self.r):
                idx = self.indices[i, j]
                sub_var_map[idx] = M_var[i, j]
        return sub_var_map

    def localizing_matrix(self, d, poly):
        """Compute the localizing matrix."""
        r = Polynomial.dim(self.n, d)
        M = self.indices[:r, :r].copy()
        for i in range(r):
            for j in range(r):
                M[i, j] = poly.localize(self.indices[i, j])
        return M

    def _compute_loc_mat_constraints(self, L_var, L_sym, sub_var_map):
        """Compute constraints to make L_var equal to variables in L_sym."""
        assert L_var.shape == L_sym.shape
        constraints = []
        for i in range(L_var.shape[0]):
            for j in range(L_var.shape[1]):
                constraints.append(
                    L_var[i, j] == L_sym[i, j].substitute(sub_var_map)
                )
        return constraints

    def moment_constraints(self, M_var):
        """Compute the constraints on a cvxpy matrix to make it a moment matrix.

        This *does not* enforce PSD constraints, however.
        """
        assert M_var.shape == self.shape
        sub_var_map = self._compute_var_map(M_var)

        constraints = []
        for i in range(self.r):
            for j in range(i, self.r):
                sub = self.indices[i, j]
                constraints.append(sub_var_map[sub] == M_var[i, j])
        return constraints

    def localizing_constraints(self, M_var, polys):
        """Add constraints on the M_d localizing matrices."""
        assert M_var.shape == self.shape

        sub_var_map = self._compute_var_map(M_var)

        constraints = []
        for p in polys:
            deg = p.degree
            if deg % 2 == 0:
                v = deg // 2
            else:
                v = (deg + 1) // 2

            # compute the symbolic localizing matrix
            L_sym = self.localizing_matrix(self.d - v, p)

            # localizing matrix variable
            L_var = cp.Variable(L_sym.shape, PSD=True)

            constraints.extend(
                self._compute_loc_mat_constraints(L_var, L_sym, sub_var_map)
            )
        return constraints

    def localize(self, M_val, poly):
        """Compute the localizing matrix for a given value of the moment matrix."""
        assert M_val.shape == self.shape

        sub_var_map = self._compute_var_map(M_val)

        deg = poly.degree
        if deg % 2 == 0:
            v = deg // 2
        else:
            v = (deg + 1) // 2

        r = Polynomial.dim(self.n, self.d - v)
        L = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                L[i, j] = poly.localize(self.indices[i, j]).substitute(sub_var_map)
        return L
