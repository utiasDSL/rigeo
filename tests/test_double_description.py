import numpy as np
from scipy.spatial.transform import Rotation

import rigeo as rg


def test_linspan():
    A = np.array([[1, 0, 0]])
    b = np.array([0])
    span_form = rg.FaceForm(A_ineq=A, b_ineq=b).to_span_form()

    # no vertices
    assert span_form.is_cone()

    # ensure linear span has been correctly mapped to rays
    assert span_form.nr == 5


def test_mixed_span_form():
    box = rg.Box(half_extents=np.ones(3))

    # box with one face cut off
    span_form = rg.FaceForm(
        A_ineq=box.A[:-1, :], b_ineq=box.b[:-1]
    ).to_span_form()

    # span form has vertices and rays
    assert not span_form.is_cone()
    assert not span_form.bounded()
    assert span_form.nv > 0
    assert span_form.nr > 0


def test_translation():
    vs = rg.box_vertices(half_extents=[1, 1, 1])
    span0 = rg.SpanForm(vertices=vs)
    face0 = span0.to_face_form()

    point = [0, 0, 0]
    assert np.all(face0.A @ point <= face0.b)

    # we want to compare the face form of the transformed span vs. the face
    # form we obtained by direct transformation
    translation = [3, 0, 0]
    span1 = span0.transform(translation=translation)
    face1 = span1.to_face_form()
    face2 = face0.transform(translation=translation)

    assert np.all(face1.A @ translation <= face1.b)
    assert np.all(face2.A @ translation <= face2.b)


def test_transform():
    vs = rg.box_vertices(half_extents=[1, 2, 1])
    span0 = rg.SpanForm(vertices=vs)
    face0 = span0.to_face_form()

    point = [1, 2, 0]
    assert np.all(face0.A @ point <= face0.b)

    # we want to compare the face form of the transformed span vs. the face
    # form we obtained by direct transformation
    translation = [3, 0, 0]
    rotation = Rotation.from_rotvec([0, 0, 0.5 * np.pi]).as_matrix()
    span1 = span0.transform(rotation=rotation, translation=translation)
    face1 = span1.to_face_form()
    face2 = face0.transform(rotation=rotation, translation=translation)

    point = rotation @ point + translation
    assert np.all(face1.A @ point <= face1.b)
    assert np.all(face2.A @ point <= face2.b)
