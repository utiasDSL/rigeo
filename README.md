# rigeo: rigid body geometry

## Features
* Build rigid bodies out of flexible shape primitives: convex polyhedra,
  ellipsoids, and cylinders.
* Check if a set of inertial parameters are physically realizable on a given
  shape using convex programming.
* Constrain a set of inertial parameters to be physically realizable during
  inertial parameter identification.
* Obtain the intersection of two convex polyhedra. This is particularly useful
  for obtaining contact patches between polyhedral objects for manipulation.
* Obtain the distance between primitive shapes using convex programming.
* Compute maximum-volume inscribed and minimum-volume bounding ellipsoids for
  sets of points.
* Compute convex hulls for degenerate sets of points (i.e., points that live in
  some lower-dimensional subspace than the ambient space).

## Usage

TODO: installation instructions

Start a virtual environment to run the scripts:
```
poetry shell
```
