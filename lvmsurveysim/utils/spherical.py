#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2017-10-20
# @Filename: spherical.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-12 17:13:11

import numpy


__all__ = ['great_circle_distance', 'ellipse_bbox', 'get_lst', 'get_altitude']


def great_circle_distance(ra0, dec0, ra1, dec1):
    """Returns the great angle distance between two points.

    Parameters:
        ra0,dec0 (float):
            The RA and Dec coordinates of the first point. In degrees.
        ra1,dec1 (float):
            The RA and Dec coordinates of the first point. In degrees.

    """

    return numpy.rad2deg(
        numpy.arccos(numpy.cos(numpy.deg2rad(dec0)) * numpy.cos(numpy.deg2rad(dec1)) *
                     numpy.cos(numpy.deg2rad(ra1 - ra0)) +
                     numpy.sin(numpy.deg2rad(dec0)) * numpy.sin(numpy.deg2rad(dec1))))


def ellipse_bbox(ra, dec, a, b, pa, padding=0):
    """Returns the bounding box in RA and Dec for a rotated ellipse.

    All parameters must be float numbers in degrees. See
    `~lvmsurveysim.target.region.EllipticalRegion` for details.

    """

    pa_rad = numpy.deg2rad(pa)

    a_x = a * numpy.sin(pa_rad)
    a_y = a * numpy.cos(pa_rad)
    b_x = b * numpy.cos(pa_rad)
    b_y = b * numpy.sin(pa_rad)

    ra_delta = numpy.sqrt(a_x**2 + b_x**2) / numpy.cos(numpy.deg2rad(dec))
    dec_delta = numpy.sqrt(a_y**2 + b_y**2)

    return (numpy.array([ra - ra_delta - padding, ra + ra_delta + padding]),
            numpy.array([dec - dec_delta - padding, dec + dec_delta + padding]))

    a_x = a * np.sin(pa_rad)
    a_y = a * np.cos(pa_rad)
    b_x = b * np.cos(pa_rad)
    b_y = b * np.sin(pa_rad)

    ra_delta = np.sqrt(a_x**2 + b_x**2) / np.cos(np.deg2rad(dec))
    dec_delta = np.sqrt(a_y**2 + b_y**2)

    return (np.array([ra - ra_delta - padding, ra + ra_delta + padding]),
            np.array([dec - dec_delta - padding, dec + dec_delta + padding]))
