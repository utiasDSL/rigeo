import numpy as np

from .shapes import ConvexPolyhedron


def Simplex(extents):
    """A :math:`d`-dimensional simplex.

    The simplex has :math:`d+1` vertices: :math:`\\boldsymbol{v}_0 =
    \\boldsymbol{0}`, :math:`\\boldsymbol{v}_1 = (e_1, 0, 0,\\dots)`,
    :math:`\\boldsymbol{v}_2 = (0, e_2, 0,\\dots)`, etc., where :math:`e_i`
    corresponds to ``extents[i]``.

    Parameters
    ----------
    extents : np.ndarray, shape (d,)
        The extents of the simplex.

    Returns
    -------
    : ConvexPolyhedron
        The simplex.
    """
    extents = np.array(extents)
    assert np.all(extents > 0), "Simplex extents must be positive."

    dim = extents.shape[0]
    vertices = np.vstack((np.zeros(dim), np.diag(extents)))
    return ConvexPolyhedron.from_vertices(vertices)
