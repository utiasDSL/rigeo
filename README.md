# rigeo: rigid body geometry

Rigeo is a prototyping library for rigid body geometry in Python: it combines
three-dimensional geometry with the inertial properties of rigid bodies, with
applications to robotic manipulation.

## Density Realizable Inertial Parameters

The main feature of this library is a set of necessary conditions for **density
realizability** on shapes which can be described as *convex hulls of
ellipsoids* (which includes convex polyhedra, cylinders, and capsules). A set
of inertial parameters (i.e., mass, center of mass, inertia matrix) is called
*density realizable* on a given shape if they can be physically realized by
*some* rigid body contained in that shape. These conditions can be included in
constraints as semidefinite programs for inertial parameter identification for
motion and force-torque data.

## Other Features

* Build rigid bodies out of flexible shape primitives: convex polyhedra,
  ellipsoids, and cylinders.
* Obtain the intersection of two convex polyhedra (via
  [cdd](https://pycddlib.readthedocs.io)). This is particularly useful for
  obtaining contact patches between polyhedral objects for manipulation; e.g.,
  when [solving the waiter's problem](https://arxiv.org/abs/2305.17484)
* Obtain the distance between primitive shapes using convex programming.
* Compute maximum-volume inscribed and minimum-volume bounding ellipsoids for
  sets of points.
* Compute convex hulls for degenerate sets of points (i.e., points that live in
  some lower-dimensional subspace than the ambient space).
* Uniform random sampling inside of and on the surface of ellipsoids.

## Installation

The library has been tested on Ubuntu 20.04 using Python 3.8; newer Python
versions and OS versions may also work. Optimization problems use
[cvxpy](https://www.cvxpy.org/); MOSEK is installed by default and is used as
the solver for the tests.

From pip:
```
pip install rigeo
```

From source (using [poetry](https://python-poetry.org)):
```
git clone https://github.com/utiasDSL/rigeo
cd rigeo
poetry shell
poetry install
# do stuff ...
```

From source (using pip):
```
git clone https://github.com/utiasDSL/rigeo
cd rigeo
python -m pip install .
```

## Development

Tests are run using pytest:
```
cd tests
python -m pytest .
```

## License

MIT - see the LICENSE file.
