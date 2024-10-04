"""Double description of convex polyhedra."""
import numpy as np
import cdd


class SpanForm:
    """Span form (V-rep) of a convex polyhedron.

    Attributes
    ----------
    vertices : np.ndarray, shape (self.nv, self.dim) or None
        The vertices of the polyhedron.
    rays : np.ndarray, shape (self.nr, self.dim) or None
        The rays of the polyhedron.
    """

    def __init__(self, vertices=None, rays=None, span=None):
        if vertices is not None:
            vertices = np.array(vertices)
        if rays is not None:
            rays = np.array(rays)

        self.vertices = vertices
        self.rays = rays

        # if we also have some linspan generators, convert these to rays
        if span is not None:
            span_rays = np.vstack((span, -span))
            if self.rays is None:
                self.rays = span_rays
            else:
                self.rays = np.vstack((self.rays, span_rays))

    def __repr__(self):
        return f"SpanForm(vertices={self.vertices}, rays={self.rays})"

    @property
    def nv(self):
        """Number of vertices."""
        return self.vertices.shape[0] if self.vertices is not None else 0

    @property
    def nr(self):
        """Number of rays."""
        return self.rays.shape[0] if self.rays is not None else 0

    @property
    def dim(self):
        """Dimension of the ambient space."""
        dim = None
        if self.vertices is not None:
            dim = self.vertices.shape[1]
        if self.rays is not None:
            dim = self.rays.shape[1]
        return dim

    @classmethod
    def from_cdd_matrix(cls, mat):
        """Construct from a CDD matrix.

        Parameters
        ----------
        mat : cdd.Matrix
            The CDD matrix representing the polyhedron.
        """
        M = np.array([mat[i] for i in range(mat.row_size)])
        if mat.row_size == 0:
            return None

        t = M[:, 0]
        v_mask = np.isclose(t, 1.0)
        r_mask = np.isclose(t, 0.0)
        s_mask = np.zeros_like(r_mask, dtype=bool)

        # handle linear spans
        lin_idx = np.array([idx for idx in mat.lin_set])
        if len(lin_idx) > 0:
            assert np.allclose(t[lin_idx], 0.0)
            s_mask[lin_idx] = True
            r_mask[lin_idx] = False

        vertices = M[v_mask, 1:] if np.any(v_mask) else None
        rays = M[r_mask, 1:] if np.any(r_mask) else None
        span = M[s_mask, 1:] if np.any(s_mask) else None
        return cls(vertices=vertices, rays=rays, span=span)

    def bounded(self):
        """Check if the polyhedron is bounded.

        Returns
        -------
        : bool
            ``True`` if the polyhedron is bounded, ``False`` otherwise."""
        return self.rays is None and self.vertices is not None

    def is_cone(self):
        """Check if the polyhedron is a cone.

        This means that for any point :math:`x` in the polyhedron, then
        :math:`\\alpha x` is also in the polyhedron for any :math:`\\alpha>0`.

        Returns
        -------
        : bool
            ``True`` if the polyhedron is a cone, ``False`` otherwise.
        """
        return self.rays is not None and self.vertices is None

    def to_cdd_matrix(self):
        """Convert to a CDD matrix.

        Returns
        -------
        : cdd.Matrix
            A CDD matrix representing the polyhedron.
        """
        n = self.nv + self.nr
        S = np.zeros((n, self.dim + 1))
        if self.vertices is not None:
            S[: self.nv, 0] = 1.0
            S[: self.nv, 1:] = self.vertices
        if self.rays is not None:
            S[self.nv :, 0] = 0.0
            S[self.nv :, 1:] = self.rays
        Smat = cdd.Matrix(S)
        Smat.rep_type = cdd.RepType.GENERATOR
        return Smat

    def canonical(self):
        """Convert to canonical non-redundant representation.

        In other words, take the convex hull of the vertices.

        Returns
        -------
        : SpanForm
            A canonicalized version of the span form.
        """
        mat = self.to_cdd_matrix()
        mat.canonicalize()
        return SpanForm.from_cdd_matrix(mat)

    def to_face_form(self):
        """Convert to face form.

        Returns
        -------
        : FaceForm
            The equivalent face form of the polyhedron.
        """
        Smat = self.to_cdd_matrix()
        poly = cdd.Polyhedron(Smat)
        Fmat = poly.get_inequalities()
        return FaceForm.from_cdd_matrix(Fmat)

    def transform(self, rotation=None, translation=None):
        if rotation is None:
            rotation = np.eye(self.dim)
        if translation is None:
            translation = np.zeros(self.dim)

        vertices = None
        rays = None
        if self.vertices is not None:
            vertices = (rotation @ self.vertices.T).T + translation
        if self.rays is not None:
            rays = (rotation @ self.rays.T).T + translation

        return SpanForm(vertices=vertices, rays=rays)


class FaceForm:
    """Face form (H-rep) of a convex polyhedron."""

    def __init__(self, A_ineq, b_ineq, A_eq=None, b_eq=None):
        # we use an inequality-only representation, where equalities are
        # represented by two-sided inequalities
        if A_eq is not None:
            assert A_eq.shape[0] == b_eq.shape[0]
            self.A = np.vstack((A_ineq, A_eq, -A_eq))
            self.b = np.concatenate((b_ineq, b_eq, -b_eq))
        else:
            self.A = A_ineq
            self.b = b_ineq

    def __repr__(self):
        return f"FaceForm(A={self.A}, b={self.b})"

    @classmethod
    def from_cdd_matrix(cls, mat):
        """Construct from a CDD matrix.

        Parameters
        ----------
        mat : cdd.Matrix
            The CDD matrix representing the polyhedron.
        """
        M = np.array([mat[i] for i in range(mat.row_size)])
        b = M[:, 0]
        A = -M[:, 1:]

        ineq_idx = np.array(
            [idx for idx in range(mat.row_size) if idx not in mat.lin_set]
        )
        eq_idx = np.array([idx for idx in mat.lin_set])

        return cls(
            A_ineq=A[ineq_idx, :],
            b_ineq=b[ineq_idx],
            A_eq=A[eq_idx, :] if len(eq_idx) > 0 else None,
            b_eq=b[eq_idx] if len(eq_idx) > 0 else None,
        )

    @property
    def nf(self):
        """Number of faces."""
        return self.A.shape[0]

    @property
    def dim(self):
        """The dimension of the ambient space."""
        return self.A.shape[1]

    def to_cdd_matrix(self):
        """Convert to a CDD matrix.

        Returns
        -------
        : cdd.Matrix
            A CDD matrix representing the polyhedron.
        """
        # face form is Ax <= b, which cdd stores as one matrix [b -A]
        F = np.hstack((self.b[:, None], -self.A))
        Fmat = cdd.Matrix(F)
        Fmat.rep_type = cdd.RepType.INEQUALITY
        return Fmat

    def canonical(self):
        """Convert to canonical non-redundant representation.

        Returns
        -------
        : FaceForm
            A canonicalized version of the face form.
        """
        mat = self.to_cdd_matrix()
        mat.canonicalize()
        return FaceForm.from_cdd_matrix(mat)

    def stack(self, other):
        """Combine two face forms together.

        The corresponds to an intersection of polyhedra.

        Parameters
        ----------
        other : FaceForm
            The other face form to combine with this one.

        Returns
        -------
        : FaceForm
            The combined face form representing the intersection.
        """
        A = np.vstack((self.A, other.A))
        b = np.concatenate((self.b, other.b))
        return FaceForm(A_ineq=A, b_ineq=b)

    def to_span_form(self):
        """Convert to span form (V-rep).

        Returns
        -------
        : SpanForm
            The equivalent span form of the polyhedron.
        """
        Fmat = self.to_cdd_matrix()
        poly = cdd.Polyhedron(Fmat)
        Smat = poly.get_generators()
        return SpanForm.from_cdd_matrix(Smat)

    def transform(self, rotation=None, translation=None):
        if rotation is None:
            rotation = np.eye(self.dim)
        if translation is None:
            translation = np.zeros(self.dim)

        A = self.A @ rotation.T
        b = self.b + A @ translation
        return FaceForm(A_ineq=A, b_ineq=b)


def convex_hull(points, rcond=None):
    """Get the vertices of the convex hull of a set of points.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
        A set of ``n`` points in ``d`` dimensions for which to compute the
        convex hull. The points do *not* need to be full rank; that is, they
        may span a lower-dimensional space than :math:`\\mathbb{R}^d`.
    rcond : float, optional
        Conditioning number used for internal routines.

    Returns
    -------
    : np.ndarray, shape (m, d)
        The vertices of the convex hull that fully contains the set of points.
    """
    assert points.ndim == 2
    if points.shape[0] <= 1:
        return points

    # qhull does not handle degenerate sets of points but cdd does, which is
    # nice
    return SpanForm(points).canonical().vertices
