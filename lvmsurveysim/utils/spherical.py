#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2017-10-20
# @Filename: spherical.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-12 17:27:27

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


def get_lst(jd, lon):
    """Returns the approximate Local Median Sidereal Time.

    Parameters
    ----------
    jd : float or ~numpy.ndarray
        The Julian Date or an array of dates.
    lon : float
        The longitude of the location.

    Returns
    -------
    lmst : float or ~numpy.ndarray
        The Local Median Sideral Time in hours. Same shape as the input ``jd``.

    """

    dd = jd - 2451545.0

    lmst = ((280.46061837 + 360.98564736629 * dd + 0.000388 *
            (dd / 36525.)**2 + lon) % 360) / 15.

    return lmst


def get_altitude(ra, dec, jd=None, lst=None, lon=None, lat=None, airmass=False):
    """Returns the altitude of an object from its equatorial coordinates.

    Parameters
    ----------
    ra : float or ~numpy.ndarray
        The Right Ascension of the object(s).
    ra : float or ~numpy.ndarray
        The declination of the object(s).
    jd : float or ~numpy.ndarray
        The Julian Date or an array of dates. The local sidereal time will
        be calculated from these dates using the longitude.
    lst : float or ~numpy.ndarray
        The local sidereal time, in hours. Overrides ``jd``.
    lon : float
        The longitude of the location.
    lat : float
        The latitude of the location.
    airmass : bool
        If `True`, returns the airmass (:math:`\sec z`) instead of the
        altitude.

    Returns
    -------
    altitude : float or ~numpy.ndarray
        The altitude of the object at the given time. Returns the airmass if
        ``airmass=True``.

    """

    ra = numpy.atleast_1d(ra)
    dec = numpy.atleast_1d(dec)

    assert len(ra) == len(dec), 'ra and dec must have the same length.'

    if jd is not None:

        assert lst is None, 'cannot set jd and lst at the same time.'

        jd = numpy.atleast_1d(jd)
        lst = get_lst(jd, lon)

    if len(lst) == 1:
        lst = numpy.repeat(lst, len(ra))
    elif len(lst) != len(ra):
        raise ValueError('jd does not have the same length as the coordinates.')

    ha = (lst * 15. - ra) % 360.

    sin_alt = (numpy.sin(numpy.radians(dec)) * numpy.sin(numpy.radians(lat)) +
               numpy.cos(numpy.radians(dec)) * numpy.cos(numpy.radians(lat)) *
               numpy.cos(numpy.radians(ha)))

    alt = numpy.rad2deg(numpy.arcsin(sin_alt))

    if airmass:
        return 1 / numpy.cos(numpy.radians(90 - alt))

    return alt
