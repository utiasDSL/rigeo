import numpy as np

import inertial_params as ip


def test_linspan():
    A = np.array([[1, 0, 0]])
    b = np.array([0])
    span_form = ip.FaceForm(A_ineq=A, b_ineq=b).to_span_form()

    # no vertices
    assert span_form.is_cone()

    # ensure linear span has been correctly mapped to rays
    assert span_form.nr == 5


def test_mixed_span_form():
    box = ip.Box(half_extents=np.ones(3))

    # box with one face cut off
    span_form = ip.FaceForm(A_ineq=box.A[:-1, :], b_ineq=box.b[:-1]).to_span_form()

    # span form has vertices and rays
    assert not span_form.is_cone()
    assert not span_form.bounded()
    assert span_form.nv > 0
    assert span_form.nr > 0
