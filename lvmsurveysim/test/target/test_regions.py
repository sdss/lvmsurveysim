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


np.random.seed(12346)


def test_shapely(region):

    assert hasattr(region, 'shapely')
    assert isinstance(region.shapely, shapely.geometry.base.BaseGeometry)


class TestCircularRegion(object):

    _region_types = ['circle']

    def _create_points(self, test_region, n_points=150, plot=False):
        """Creates a list of points within the bounds of the test region.

        Points are determined to be inside or outside the region by calculating
        the spherical distance to the centre of the region.

        """

        ra0 = test_region.coords.ra.deg
        dec0 = test_region.coords.dec.deg

        ra_bounds = test_region.r.deg / np.cos(np.deg2rad(dec0))
        dec_bounds = test_region.r.deg

        test_points_ra = ra0 + 2 * ra_bounds * np.random.sample(n_points) - ra_bounds
        test_points_dec = dec0 + 2 * dec_bounds * np.random.sample(n_points) - dec_bounds

        test_points = np.array([test_points_ra, test_points_dec]).T

        sph_distance = np.arccos(np.cos(np.deg2rad(dec0)) *
                                 np.cos(np.deg2rad(test_points_dec)) *
                                 np.cos(np.deg2rad(test_points_ra - ra0)) +
                                 np.sin(np.deg2rad(dec0)) * np.sin(np.deg2rad(test_points_dec)))
        sph_distance = np.rad2deg(sph_distance)

        inside_points = test_points[np.where(sph_distance < test_region.r.deg)]
        outside_points = test_points[np.where(sph_distance >= test_region.r.deg)]

        if plot:
            fig, ax = test_region.plot(fill=False, edgecolor='k', linewidth=1.5)
            ax.scatter(inside_points[:, 0], inside_points[:, 1], marker='o', edgecolor='None')
            ax.scatter(outside_points[:, 0], outside_points[:, 1], marker='o', edgecolor='None')
            image_path = str(pathlib.Path(__file__).parents[1] / 'plots/test_circular_region.png')
            plt.savefig(str(image_path))

        return inside_points, outside_points

    def _plot_points_shapely(self, region, points):
        """Plots a list of points on the shapely region."""

        shapely_points = [(point[0], point[1]) for point in points]

        shapely_inside = np.array(list(filter(region.contains, shapely_points)))
        shapely_outside = np.array(list(filter(lambda xx: not region.contains(xx),
                                               shapely_points)))

        fig, ax = region.plot(fill=False, edgecolor='k', linewidth=1.5)

        ax.scatter(shapely_inside[:, 0], shapely_inside[:, 1], marker='o', edgecolor='None')
        ax.scatter(shapely_outside[:, 0], shapely_outside[:, 1], marker='o', edgecolor='None')

        image_path = str(pathlib.Path(__file__).parents[1] /
                         'plots/test_circular_region_shapely.png')

        plt.savefig(str(image_path))

    def test_point_inside_ellipse(self, region, plot):

        inside_points, outside_points = self._create_points(region, plot=plot)

        if plot:
            all_points = np.vstack((inside_points, outside_points))
            self._plot_points_shapely(region, all_points)

        for point in inside_points:
            assert region.contains(point)

        for point in outside_points:
            assert not region.contains(point)
