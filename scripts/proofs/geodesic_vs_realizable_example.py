"""This example shows that the geodesic distance does not follow along with
realizability."""
import numpy as np

import inertial_params as ip

import IPython


def main():
    m0 = 1
    r0 = 0.1
    H0 = m0 * r0**2 * np.eye(3) / 3
    params0 = ip.RigidBody(mass=m0, h=np.zeros(3), H=H0)
    ellipsoid0 = ip.Ellipsoid.sphere(radius=r0)

    assert np.trace(params0.J @ ellipsoid0.Q) >= 0

    # second parameters only realizable on second ellipsoid, not the first
    r1 = 2 * r0
    # H1 = m0 * r1**2 * np.eye(3) / 3
    # params1 = ip.RigidBody(mass=m0, h=np.zeros(3), H=H1)
    ellipsoid1 = ip.Ellipsoid.sphere(radius=r1)

    # construct "flattish" parameters"
    masses = m0 * np.ones(6) / 6
    points = r1 * np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0.1, 0],
            [0, -0.1, 0],
            [0, 0, 0.1],
            [0, 0, -0.1],
        ]
    )
    params1 = ip.RigidBody.from_point_masses(masses, points)

    assert np.trace(params1.J @ ellipsoid1.Q) >= 0

    r2 = 3 * r0
    H2 = m0 * r2**2 * np.eye(3) / 3
    params2 = ip.RigidBody(mass=m0, h=np.zeros(3), H=H2)

    # the key is that this is *not* realizable on the ellipsoid but is *closer*
    # (in terms of geodesic distance) to the inner J0
    assert np.trace(ellipsoid1.Q @ params2.J) < 0

    d01 = ip.positive_definite_distance(params0.J, params1.J)
    d02 = ip.positive_definite_distance(params0.J, params2.J)

    print(f"d01 = {d01}")
    print(f"d02 = {d02}")

    # print(np.trace(ellipsoid0.Q @ params0.J))
    # print(np.trace(ellipsoid1.Q @ params0.J))
    # print(np.trace(ellipsoid0.Q @ params1.J))
    # print(np.trace(ellipsoid1.Q @ params2.J))


main()
