#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 20, 2017
# @Filename: spherical.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


__all__ = ['great_circle_distance', 'ellipse_bbox']


def great_circle_distance(ra0, dec0, ra1, dec1):
    """Returns the great angle distance between two points.

    Parameters:
        ra0,dec0 (float):
            The RA and Dec coordinates of the first point. In degrees.
        ra1,dec1 (float):
            The RA and Dec coordinates of the first point. In degrees.

    """

    return np.rad2deg(
        np.arccos(np.cos(np.deg2rad(dec0)) * np.cos(np.deg2rad(dec1)) *
                  np.cos(np.deg2rad(ra1 - ra0)) +
                  np.sin(np.deg2rad(dec0)) * np.sin(np.deg2rad(dec1))))


def ellipse_bbox(ra, dec, a, b, pa, padding=0):
    """Returns the bounding box in RA and Dec for a rotated ellipse.

    All parameters must be float numbers in degrees. See
    `~lvmsurveysim.target.regions.EllipticalRegion` for details.

    """

    pa_rad = np.deg2rad(pa)

    a_x = a * np.sin(pa_rad)
    a_y = a * np.cos(pa_rad)
    b_x = b * np.cos(pa_rad)
    b_y = b * np.sin(pa_rad)

    ra_delta = np.sqrt(a_x**2 + b_x**2) / np.cos(np.deg2rad(dec))
    dec_delta = np.sqrt(a_y**2 + b_y**2)

    return (np.array([ra - ra_delta - padding, ra + ra_delta + padding]),
            np.array([dec - dec_delta - padding, dec + dec_delta + padding]))
