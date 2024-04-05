#!/usr/bin/env python3
import argparse

import wrlparser
import numpy as np

import rigeo as rg

import IPython


NUM_POINTS_PER_POLY = 5
BB_DIAG_LEN = 1


# TODO probably want some logic to scale the bounding box diagonal to some
# fixed size, like 1 meter
class WRL:
    """Parse polyhedrons from a WRL/VRML file."""

    def __init__(self, data):
        points = []
        for shape in data.nodes:
            points.append(shape.geometry.coord.point)
        points = np.vstack(points)

        # bounding box
        box = rg.Box.from_points_to_bound(points)

        diaglen = 2 * np.linalg.norm(box.half_extents)
        scale = BB_DIAG_LEN / diaglen

        # TODO do I need to shift to origin? maybe I should be using the points
        # centroid?
        # TODO need to rescale all the points of the individual polyhedra
        points0 = points - box.center
        points0_scaled = scale * points
        points_scaled = points0_scaled + box.center

        import IPython
        IPython.embed()

        self.nv = 0
        self.polys = []

        for shape in data.nodes:
            points = shape.geometry.coord.point
            poly = rg.ConvexPolyhedron.from_vertices(points)
            self.nv += len(points)
            self.polys.append(poly)

    @classmethod
    def from_string(cls, s):
        """Parse from a string."""
        data = wrlparser.parse(s)
        return cls(data)

    @classmethod
    def from_file_path(cls, path):
        """Parse from a file."""
        with open(path) as f:
            s = f.read()
        return cls.from_string(s)

    def random_points(self, n):
        """Generate random points contained in the shapes in the scene."""
        # uniformly randomly choose which shapes to generate points in
        num_poly = len(self.polys)
        prob = np.ones(num_poly) / num_poly  # uniform
        num_per_poly = np.random.multinomial(n, prob)

        # generate the random points in each polyhedron
        points = []
        for num, poly in zip(num_per_poly, self.polys):
            points.append(poly.random_points(num))
        return np.vstack(points)


def main():
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("wrlfile", help="WRL/VRML file to load.")
    args = parser.parse_args()

    scene = WRL.from_file_path(args.wrlfile)

    # TODO need to generate random masses inside and see how it goes
    # we have one set of random parameters for the whole shape, but we compose
    # these together in the optimization problem to identify it
    points = scene.random_points(20)

    IPython.embed()


main()
