import numpy as np

import inertial_params as ip


def main():
    cuboid_half_extents = 0.5 * np.array([0.1, 0.1, 0.4])
    box = ip.Box(cuboid_half_extents)
    ell_outer = box.minimum_bounding_ellipsoid()
    ell_inner = box.maximum_inscribed_ellipsoid()
    print(f"Bounding ellipsoid half extents = {ell_outer.half_extents}")
    print(f"Inscribed ellipsoid half extents = {ell_inner.half_extents}")

    params = box.vertex_point_mass_params(mass=1.0)
    print(params.I)

    import IPython
    IPython.embed()


main()
