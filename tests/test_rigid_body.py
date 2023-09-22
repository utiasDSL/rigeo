import numpy as np
import inertial_params as ip


def test_rigid_body_addition():
    mass = 1.0
    h = np.zeros(3)
    I = ip.cuboid_inertia_matrix(mass=mass, half_extents=[0.5, 0.5, 0.5])
    H = ip.I2H(I)

    b1 = ip.RigidBody(mass=mass, h=h, H=H)
    b_sum = b1 + b1

    assert np.allclose(b_sum.J, 2 * b1.J)
    assert np.allclose(b_sum.I, 2 * b1.I)
    assert np.allclose(b_sum.θ, 2 * b1.θ)

    # more complex example with non-zero h
    h = np.array([1, 2, 3])
    H = H + np.outer(h, h) / mass
    b2 = ip.RigidBody(mass=mass, h=h, H=H)
    b_sum = b1 + b2

    assert np.allclose(b_sum.J, b1.J + b2.J)
    assert np.allclose(b_sum.I, b1.I + b2.I)
    assert np.allclose(b_sum.θ, b1.θ + b2.θ)


def test_rigid_body_representations():
    mass = 1.0
    h = np.array([1, 2, 3])
    Ic = ip.cuboid_inertia_matrix(mass=mass, half_extents=[0.5, 0.5, 0.5])
    H = ip.I2H(Ic) + np.outer(h, h) / mass

    b1 = ip.RigidBody(mass=mass, h=h, H=H)

    b2 = ip.RigidBody.from_vector(b1.θ)
    b3 = ip.RigidBody.from_pseudo_inertia_matrix(b1.J)
    b4 = ip.RigidBody.from_mcI(mass=b1.mass, com=b1.com, I=b1.I)

    assert np.allclose(b2.J, b1.J)
    assert np.allclose(b3.J, b1.J)
    assert np.allclose(b4.J, b1.J)

