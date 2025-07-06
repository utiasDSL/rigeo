# rigeo: rigid body geometry and inertial parameters

rigeo is a prototyping library for working with rigid bodies in Python: it combines
three-dimensional geometry with the inertial properties of rigid bodies, with
applications to robotic manipulation.

## Density Realizable Inertial Parameters

One of the main features of this library is a set of necessary conditions for
**density realizability** on 3D shapes based on moment relaxations (see [this
paper](https://arxiv.org/abs/2411.07079) for more information). A set of
inertial parameters (i.e., mass, center of mass, inertia matrix) is called
*density realizable* on a given shape if it can be physically realized by
*some* rigid body contained in that shape. These conditions can be included as
constraints in semidefinite programs for inertial parameter identification or
for checking robustness to inertial parameter uncertainty.

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

The library requires Python >=3.8. It has been tested on Ubuntu 20.04 and
24.04. Optimization problems use [cvxpy](https://www.cvxpy.org/); MOSEK is
installed by default and is used as the solver for the tests. Academic licenses
for MOSEK can be obtained for free. If this is not an option for you, Clarabel
is a reasonable open-source alternative.

From pip:
```
pip install rigeo
```

From source (using [uv](https://docs.astral.sh/uv/)):
```
git clone https://github.com/utiasDSL/rigeo
cd rigeo
uv venv
uv sync
```

From source (using pip):
```
git clone https://github.com/utiasDSL/rigeo
cd rigeo
python -m pip install .
```

## Scripts

You can find a variety of scripts in the `scripts` directory:

* `examples`: numerical examples of approximating inertial parameters for
  particular bodies using random sampling (see [this blog
  post](https://adamheins.com/blog/cuboid-inertia));
* `experiments`: numerical experiments comparing density realizability
  constraints based on moment relaxations to custom constraints for boxes and
  cylinders (the custom ones yield the same results but result in much faster
  semidefinite programs);
* `theory`: symbolic derivations of some inertial parameter results.

## Development

Tests are run using pytest:
```
cd tests
python -m pytest .
```

To test against different Python versions, use:
```
# for example
uv run --isolated --python=3.9 pytest
```

## License

MIT - see the LICENSE file.
