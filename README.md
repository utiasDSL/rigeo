# rigeo: rigid body geometry

Rigeo is a prototyping library for rigid body geometry in Python: it combines
three-dimensional geometry with the inertial properties of rigid bodies, with
applications to robotic manipulation.

## Density Realizable Inertial Parameters

The main feature of this library is a set of necessary conditions for **density
realizability** on various primitive shapes (included convex polyhedra,
cylinders, and capsules). A set of inertial parameters is called density
realizable on a given shape if they can be physically realized by some rigid
body contained in that shape. These conditions can be included in constraints
as semidefinite programs for inertial parameter identification for motion and
force-torque data.

## Other Features

* Build rigid bodies out of flexible shape primitives: convex polyhedra,
  ellipsoids, and cylinders.
* Obtain the intersection of two convex polyhedra. This is particularly useful
  for obtaining contact patches between polyhedral objects for manipulation;
  e.g., when [solving the waiter's problem](https://arxiv.org/abs/2305.17484)
* Obtain the distance between primitive shapes using convex programming.
* Compute maximum-volume inscribed and minimum-volume bounding ellipsoids for
  sets of points.
* Compute convex hulls for degenerate sets of points (i.e., points that live in
  some lower-dimensional subspace than the ambient space).

## Installation

TODO: pip installation, installation instructions

## Usage

Start a virtual environment to run the scripts:
```
poetry shell
```

