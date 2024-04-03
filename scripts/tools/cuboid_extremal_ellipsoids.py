import numpy as np

import rigeo as rg


MASS = 1.0

def main():
    box = rg.Box([0.0525, 0.0525, 0.4])
    ell_outer = box.mbe()
    ell_inner = box.mie()
    print(f"Bounding ellipsoid half extents = {ell_outer.half_extents}")
    print(f"Inscribed ellipsoid half extents = {ell_inner.half_extents}")

    # params = box.vertex_point_mass_params(mass=MASS)
    # print(params.I)
    #
    # # to get simulated actual inertia about the CoM
    # box2 = rg.Box([0.02, 0.02, 0.08])
    # params2 = box2.vertex_point_mass_params(mass=MASS)
    # print(params2.I)
    #
    # import IPython
    # IPython.embed()


main()
