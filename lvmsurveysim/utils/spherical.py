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


__all__ = ['great_circle_distance']


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
