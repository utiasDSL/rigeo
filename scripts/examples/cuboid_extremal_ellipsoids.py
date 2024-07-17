"""Compute the minimum-volume bounding ellipsoid and maximum-volume inscribed
ellipsoid of a given box."""
import rigeo as rg


def main():
    box = rg.Box([1.0, 0.5, 0.5])
    mbe = box.mbe()  # max-volume bounding
    mie = box.mie()  # min-volume inscribed

    print(f"Box half extents = {box.half_extents}")
    print(f"Bounding ellipsoid half extents = {mbe.half_extents}")
    print(f"Inscribed ellipsoid half extents = {mie.half_extents}")


main()
