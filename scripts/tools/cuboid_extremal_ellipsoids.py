import numpy as np

import inertial_params as ip


def main():
    cuboid_half_extents = 0.5 * np.array([0.1, 0.1, 0.4])
    box = ip.AxisAlignedBox(cuboid_half_extents)
    ell_outer = ip.minimum_bounding_ellipsoid(box.vertices)
    ell_inner = ip.maximum_inscribed_ellipsoid(box.vertices)

    r1, V1 = ell_outer.axes()
    print(f"Bounding ellipsoid half extents = {r1}")

    r2, V2 = ell_inner.axes()
    print(f"Inscribed ellipsoid half extents = {r2}")


main()
