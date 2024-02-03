import numpy as np
import inertial_params as ip


def test_inertial_params_addition():
    mass = 1.0
    h = np.zeros(3)
    I = ip.Box(half_extents=[0.5, 0.5, 0.5]).uniform_density_params(mass).I
    H = ip.I2H(I)

    p1 = ip.InertialParameters(mass=mass, h=h, H=H)
    p_sum = p1 + p1

    assert np.allclose(p_sum.J, 2 * p1.J)
    assert np.allclose(p_sum.I, 2 * p1.I)
    assert np.allclose(p_sum.θ, 2 * p1.θ)

    # more complex example with non-zero h
    h = np.array([1, 2, 3])
    H = H + np.outer(h, h) / mass
    p2 = ip.InertialParameters(mass=mass, h=h, H=H)
    p_sum = p1 + p2

    assert np.allclose(p_sum.J, p1.J + p2.J)
    assert np.allclose(p_sum.I, p1.I + p2.I)
    assert np.allclose(p_sum.θ, p1.θ + p2.θ)


def test_inertial_params_representations():
    mass = 1.0
    h = np.array([1, 2, 3])
    Ic = ip.Box(half_extents=[0.5, 0.5, 0.5]).uniform_density_params(mass).I
    H = ip.I2H(Ic) + np.outer(h, h) / mass

    p1 = ip.InertialParameters(mass=mass, h=h, H=H)
    p2 = ip.InertialParameters.from_vector(p1.θ)
    p3 = ip.InertialParameters.from_pseudo_inertia_matrix(p1.J)
    p4 = ip.InertialParameters.from_mcI(mass=p1.mass, com=p1.com, I=p1.I)

    assert np.allclose(p2.J, p1.J)
    assert np.allclose(p3.J, p1.J)
    assert np.allclose(p4.J, p1.J)

