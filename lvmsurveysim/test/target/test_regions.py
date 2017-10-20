#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 19, 2017
# @Filename: test_regions.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pathlib

import matplotlib.pyplot as plt
import numpy as np

import shapely.geometry

from lvmsurveysim.target.regions import CircularRegion
from lvmsurveysim.utils.spherical import great_circle_distance


# Sets a fixed seed for the random numbers, so that results are reproducible.
np.random.seed(12346)


def plot_inside_outside(region, inside_points, outside_points, plot_fn):
    """Plots the region and the inside and outside points."""

    image_path = str(pathlib.Path(__file__).parents[1] / 'plots' / plot_fn)

    fig, ax = region.plot(fill=False, edgecolor='k', linewidth=1.5)
    ax.scatter(inside_points[:, 0], inside_points[:, 1], marker='o', edgecolor='None')
    ax.scatter(outside_points[:, 0], outside_points[:, 1], marker='o', edgecolor='None')
    plt.savefig(str(image_path))


def get_random_points(low, high, nn=150):
    """Returns points from the uniform distribution between low and high."""

    return (high - low) * np.random.random(nn) + low


def test_shapely(region):

    assert hasattr(region, 'shapely')
    assert isinstance(region.shapely, shapely.geometry.base.BaseGeometry)


class TestCircularRegion(object):

    _region_types = ['circle']

    def _get_bounds(self, test_region):
        """Manually calculates the bounds of the region."""

        ra0 = test_region.coords.ra.deg
        dec0 = test_region.coords.dec.deg

        ra_size = test_region.r.deg / np.cos(np.deg2rad(dec0))
        dec_size = test_region.r.deg

        return (np.array([ra0 - ra_size, ra0 + ra_size]),
                np.array([dec0 - dec_size, dec0 + dec_size]))

    def _create_points(self, test_region, n_points=150, plot=False):
        """Creates a list of points within the bounds of the test region.

        Points are determined to be inside or outside the region by calculating
        the spherical distance to the centre of the region.

        """

        ra0 = test_region.coords.ra.deg
        dec0 = test_region.coords.dec.deg

        ra_bounds, dec_bounds = self._get_bounds(test_region)

        test_points_ra = get_random_points(ra_bounds[0], ra_bounds[1], nn=n_points)
        test_points_dec = get_random_points(dec_bounds[0], dec_bounds[1], nn=n_points)

        sph_distance = great_circle_distance(test_points_ra, test_points_dec, ra0, dec0)

        test_points = np.array([test_points_ra, test_points_dec]).T
        inside_points = test_points[np.where(sph_distance < test_region.r.deg)]
        outside_points = test_points[np.where(sph_distance >= test_region.r.deg)]

        if plot:
            plot_inside_outside(test_region, inside_points, outside_points,
                                'test_circular_region.png')

        return inside_points, outside_points

    def _plot_points_shapely(self, region, points):
        """Plots a list of points on the shapely region."""

        shapely_points = [(point[0], point[1]) for point in points]

        shapely_inside = np.array(list(filter(region.contains, shapely_points)))
        shapely_outside = np.array(list(filter(lambda xx: not region.contains(xx),
                                               shapely_points)))

        plot_inside_outside(region, shapely_inside, shapely_outside,
                            'test_circular_region_shapely.png')

    def test_point_inside_ellipse(self, region, plot):

        inside_points, outside_points = self._create_points(region, plot=plot)

        if plot:
            all_points = np.vstack((inside_points, outside_points))
            self._plot_points_shapely(region, all_points)

        for point in inside_points:
            assert region.contains(point)

        for point in outside_points:
            assert not region.contains(point)

    def test_bases(self, region):

        assert isinstance(region, CircularRegion)
