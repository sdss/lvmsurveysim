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

import abc

import pathlib

import matplotlib.pyplot as plt
import numpy as np

import shapely.geometry

import lvmsurveysim.target.regions
from lvmsurveysim.utils.spherical import great_circle_distance, ellipse_bbox


# Sets a fixed seed for the random numbers, so that results are reproducible.
np.random.seed(12346)


def get_random_points(low, high, nn=150):
    """Returns points from the uniform distribution between low and high."""

    return (high - low) * np.random.random(nn) + low


def test_shapely(region):

    assert hasattr(region, 'shapely')
    assert isinstance(region.shapely, shapely.geometry.base.BaseGeometry)


class RegionBaseTester(object, metaclass=abc.ABCMeta):
    """A base class for all region testing."""

    _region_type = ''
    _region_class = None

    # Sometimes shapely's criterion for whether a point is inside or outside is different
    # from the calculation. This happens always in very edge cases. This allows to set the
    # number of points that can fail.
    _max_failures = 0

    @abc.abstractmethod
    def _get_bounds(self, test_region):
        """Manually calculates the bounds of the region."""

        pass

    @abc.abstractmethod
    def _create_points(self, test_region, n_points=150):
        """Creates a list of points within the bounds of the test region.

        Points are determined to be inside or outside the region by calculating
        the spherical distance to the centre of the region.

        """

        pass

    def _plot_inside_outside(self, region, inside_points, outside_points, plot_fn):
        """Plots the region and the inside and outside points."""

        image_path = str(pathlib.Path(__file__).parents[1] / 'plots' / plot_fn)

        fig, ax = region.plot(fill=False, edgecolor='k', linewidth=1.5)
        ax.scatter(inside_points[:, 0], inside_points[:, 1], marker='o', edgecolor='None')
        ax.scatter(outside_points[:, 0], outside_points[:, 1], marker='o', edgecolor='None')

        plt.savefig(str(image_path))

    def _plot_points_shapely(self, region, points):
        """Plots a list of points on the shapely region."""

        shapely_points = [(point[0], point[1]) for point in points]

        shapely_inside = np.array(list(filter(region.contains, shapely_points)))
        shapely_outside = np.array(list(filter(lambda xx: not region.contains(xx),
                                               shapely_points)))

        self._plot_inside_outside(region, shapely_inside, shapely_outside,
                                  'test_' + region.name + '_shapely.pdf')

    def test_self(self):

        assert len(self._region_type) > 0, '_region_type not defined.'
        assert self._region_class is not None, '_region_class not defined.'

    def test_point_inside(self, region, plot):

        inside_points, outside_points = self._create_points(region)

        if plot:

            self._plot_inside_outside(region, inside_points, outside_points,
                                      'test_' + region.name + '_region.pdf')

            all_points = np.vstack((inside_points, outside_points))
            self._plot_points_shapely(region, all_points)

        n_failures = 0

        for point in inside_points:
            if not region.contains(point):
                n_failures += 1

        for point in outside_points:
            if region.contains(point):
                n_failures += 1

        assert n_failures <= self._max_failures, f'shapely failed to classify {n_failures} points.'

    def test_bases(self, region):

        assert isinstance(region, self._region_class)


class TestCircularRegion(RegionBaseTester):

    _region_type = 'circle'
    _region_class = lvmsurveysim.target.regions.CircularRegion
    _plot_fn_base = 'test_circular'

    _max_failures = 1

    def _get_bounds(self, test_region, padding=0):
        """Manually calculates the bounds of the region."""

        ra0 = test_region.coords.ra.deg
        dec0 = test_region.coords.dec.deg

        ra_size = (test_region.r.deg + padding) / np.cos(np.deg2rad(dec0))
        dec_size = test_region.r.deg + padding

        return (np.array([ra0 - ra_size, ra0 + ra_size]),
                np.array([dec0 - dec_size, dec0 + dec_size]))

    def _create_points(self, test_region, n_points=150, plot=False):
        """Creates a list of points within the bounds of the test region.

        Points are determined to be inside or outside the region by calculating
        the spherical distance to the centre of the region.

        """

        ra0 = test_region.coords.ra.deg
        dec0 = test_region.coords.dec.deg

        ra_bounds, dec_bounds = self._get_bounds(test_region, padding=0.02)

        test_points_ra = get_random_points(ra_bounds[0], ra_bounds[1], nn=n_points)
        test_points_dec = get_random_points(dec_bounds[0], dec_bounds[1], nn=n_points)

        sph_distance = great_circle_distance(test_points_ra, test_points_dec, ra0, dec0)

        test_points = np.array([test_points_ra, test_points_dec]).T
        inside_points = test_points[np.where(sph_distance < test_region.r.deg)]
        outside_points = test_points[np.where(sph_distance >= test_region.r.deg)]

        return inside_points, outside_points


class TestEllipticalRegion(RegionBaseTester):

    _region_type = 'ellipse'
    _region_class = lvmsurveysim.target.regions.EllipticalRegion
    _plot_fn_base = 'test_elliptical'

    _max_failures = 3

    def _get_bounds(self, test_region, padding=0):
        """Manually calculates the bounds of the region."""

        ra0 = test_region.coords.ra.deg
        dec0 = test_region.coords.dec.deg

        a = test_region.a.deg
        b = test_region.b.deg
        pa = test_region.pa.deg

        return ellipse_bbox(ra0, dec0, a, b, pa, padding=padding)

    def _create_points(self, test_region, n_points=600, plot=False):
        """Creates a list of points within the bounds of the test region.

        Points are determined to be inside or outside the region by calculating
        the spherical distance to the centre of the region.

        """

        ra0 = test_region.coords.ra.deg
        dec0 = test_region.coords.dec.deg

        ra_bounds, dec_bounds = self._get_bounds(test_region, padding=0.0)

        test_points_ra = get_random_points(ra_bounds[0], ra_bounds[1], nn=n_points)
        test_points_dec = get_random_points(dec_bounds[0], dec_bounds[1], nn=n_points)

        ra_sep = (test_points_ra - ra0) * np.cos(np.deg2rad(dec0))
        dec_sep = test_points_dec - dec0
        pa_rad = np.deg2rad(-test_region.pa.deg)

        dist = (ra_sep * np.sin(pa_rad) - dec_sep * np.cos(pa_rad))**2 / test_region.a.deg**2
        dist += (ra_sep * np.cos(pa_rad) + dec_sep * np.sin(pa_rad))**2 / test_region.b.deg**2

        test_points = np.array([test_points_ra, test_points_dec]).T
        inside_points = test_points[np.where(dist < 1)]
        outside_points = test_points[np.where(dist >= 1)]

        return inside_points, outside_points


class TestPolygonalRegion(RegionBaseTester):

    _region_type = 'polygon'
    _region_class = lvmsurveysim.target.regions.PolygonalRegion
    _plot_fn_base = 'test_polygon'

    _max_failures = 1

    def _get_bounds(self, test_region, padding=0):
        """Manually calculates the bounds of the region."""

        bounds = test_region.shapely.bounds

        return (np.array([bounds[0] - padding, bounds[2] + padding]),
                np.array([bounds[1] - padding, bounds[3] + padding]))

    def _create_points(self, test_region, n_points=150, plot=False):
        """Creates a list of points within the bounds of the test region."""

        ra_bounds, dec_bounds = self._get_bounds(test_region, padding=0.02)

        test_points_ra = get_random_points(ra_bounds[0], ra_bounds[1], nn=n_points)
        test_points_dec = get_random_points(dec_bounds[0], dec_bounds[1], nn=n_points)

        test_points = np.array([test_points_ra, test_points_dec]).T

        return test_points

    def test_point_inside(self, region, plot):

        # TODO: we actually don't test anything here, just plot the points
        # according to Shapely. At some point it'd be good to have some
        # independent check of the inside-outside of these points using the
        # ray algorightm.

        points = self._create_points(region)

        if plot:

            self._plot_points_shapely(region, points)

        n_failures = 0

        assert n_failures <= self._max_failures, f'shapely failed to classify {n_failures} points.'
