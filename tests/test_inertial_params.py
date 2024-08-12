import numpy as np
from spatialmath.base import rotz

import rigeo as rg


def test_addition():
    mass = 1.0
    h = np.zeros(3)
    I = rg.Box(half_extents=[0.5, 0.5, 0.5]).uniform_density_params(mass).I
    H = rg.I2H(I)

    p1 = rg.InertialParameters(mass=mass, h=h, H=H)
    p_sum = p1 + p1

    assert np.allclose(p_sum.J, 2 * p1.J)
    assert np.allclose(p_sum.I, 2 * p1.I)
    assert np.allclose(p_sum.vec, 2 * p1.vec)

    # more complex example with non-zero h
    h = np.array([1, 2, 3])
    H = H + np.outer(h, h) / mass
    p2 = rg.InertialParameters(mass=mass, h=h, H=H)
    p_sum = p1 + p2

    assert np.allclose(p_sum.J, p1.J + p2.J)
    assert np.allclose(p_sum.I, p1.I + p2.I)
    assert np.allclose(p_sum.vec, p1.vec + p2.vec)


def test_representations():
    mass = 1.0
    h = np.array([1, 2, 3])
    Hc = rg.Box(half_extents=[1, 0.5, 0.25]).uniform_density_params(mass).H
    H = Hc + np.outer(h, h) / mass

    p1 = rg.InertialParameters(mass=mass, h=h, H=H)
    p2 = rg.InertialParameters.from_vec(p1.vec)
    p3 = rg.InertialParameters.from_pim(p1.J)
    p4 = rg.InertialParameters(mass=p1.mass, com=p1.com, I=p1.I)

    assert p2.is_same(p1)
    assert p3.is_same(p1)
    assert p4.is_same(p1)


def test_pim_sum_vec_matrices():
    rng = np.random.default_rng(0)
    As = rg.pim_sum_vec_matrices()

    for _ in range(10):
        params = rg.InertialParameters.random(rng=rng)
        J = sum([A * p for A, p in zip(As, params.vec)])
        assert np.allclose(J, params.J)


def test_transform():
    mass = 1.0
    box = rg.Box(half_extents=np.ones(3))
    params = box.vertex_point_mass_params(mass)

    # rotation only
    C = rotz(np.pi / 4)
    box2 = box.transform(rotation=C)
    params2 = params.transform(rotation=C)
    assert params2.is_same(box2.vertex_point_mass_params(mass))

    # rotation and translation
    C = rotz(np.pi / 4)
    r = np.array([1, 0.5, 0])
    box3 = box.transform(rotation=C, translation=r)
    params3 = params.transform(rotation=C, translation=r)
    assert params3.is_same(box3.vertex_point_mass_params(mass))
