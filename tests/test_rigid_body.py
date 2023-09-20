import numpy as np
import inertial_params as ip


def test_rigid_body_addition():
    mass = 1.0
    com = np.zeros(3)
    I = ip.cuboid_inertia_matrix(mass=mass, half_extents=[0.5, 0.5, 0.5])

    b1 = ip.RigidBody(mass=mass, com=com, I=I)
    b2 = b1 + b1

    assert np.allclose(b2.J, 2 * b1.J)
    assert np.allclose(b2.I, 2 * b1.I)
    assert np.allclose(b2.θ, 2 * b1.θ)
