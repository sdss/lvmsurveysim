#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-03-06
# @Filename: plan.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-13 13:15:57

import datetime
import warnings

import astral
import astropy
import astropy.coordinates
import numpy


__all__ = ['ObservingPlan', 'get_moon_data']


_delta_dt = datetime.timedelta(days=1)


def get_moon_data(jd, location=None):
    """Computes the Moon position and illuminated fraction for a given JD(s).

    Adapted from PyAstronomy.

    Parameters
    ----------
    jd : float or ~numpy.ndarray
        The Julian date.
    location : ~astropy.coordinates.EarthLocation
        The location of the observation. If `None`, assumes a geocentric
        position.

    Returns
    -------
    data : `tuple`
        A tuple with the `~astropy.coordinates.SkyCoord` position of the Moon
        and the illuminated fraction [0 - 1]. Has the same size as ``jd``.

    """

    if not isinstance(jd, astropy.time.Time):
        times = astropy.time.Time(numpy.array(jd, ndmin=1), format='jd')
    else:
        times = jd

    # Earth-Sun distance (1 AU)
    edist = 1.49598e8

    mpos = astropy.coordinates.get_moon(times, location=location)
    ram = mpos.ra.deg * numpy.pi / 180.
    decm = mpos.dec.deg * numpy.pi / 180.
    dism = mpos.distance.km

    spos = astropy.coordinates.get_sun(times)
    ras = spos.ra.deg * numpy.pi / 180.
    decs = spos.dec.deg * numpy.pi / 180.

    phi = numpy.arccos(numpy.sin(decs) * numpy.sin(decm) +
                       numpy.cos(decs) * numpy.cos(decm) * numpy.cos(ras - ram))
    inc = numpy.arctan2(edist * numpy.sin(phi), dism - edist * numpy.cos(phi))
    k = (1 + numpy.cos(inc)) / 2.

    return mpos, numpy.ravel(k)


class ObservingPlan(object):
    """Creates an observing plan.

    Parameters
    ----------
    start : float or path-like
        Either the JD of the start date.
    end : float
        The JD of the end date.
    format : str
        The format of the start and end date. Must be a format that
        `astropy.time.Time` can understand.
    observatory : str
        The observatory, either ``'APO'`` or ``'LCO'``.
    summer_shutdown : list
        A list of JDs that correspond to summer shutdown. Those JDs will be
        excluded from the range between ``(start, end)``.
    twilight_alt : float
        The altitude in degrees at which to consider that the twilight has
        began or finished. A positive value although it corresponds to a
        depression below the horizon.

    Attributes
    ----------
    data : ~astropy.table.Table
        A `~astropy.table.Table` with the schedule information including
        twilight times, Moon position, and Moon phase.

    """

    def __init__(self, start, end=None, format='jd', observatory='APO',
                 summer_shutdown=None, twilight_alt=15):

        if observatory == 'APO':
            full_obs_name = 'Apache Point Observatory'
        elif observatory == 'LCO':
            full_obs_name = 'Las Campanas Observatory'
        else:
            raise ValueError(f'invalid observatory {observatory!r}.')

        self.observatory = observatory
        self.location = astropy.coordinates.EarthLocation.of_site(full_obs_name)

        self._astral = astral.Astral()

        if isinstance(start, astropy.table.Table):

            self.data = start
            self.twilight_alt = twilight_alt

            return

        self.twilight_alt = twilight_alt

        start_date = astropy.time.Time(start, format=format)
        end_date = astropy.time.Time(end, format=format)

        jds = [jd for jd in range(int(start_date.jd), int(end_date.jd) + 1)
               if summer_shutdown is None or jd not in summer_shutdown]

        # astropy.time.Time has a lot of overhead when called so we want to
        # always do the conversions between times as an array and not as a
        # list comprehension.
        ap_times = astropy.time.Time(jds, format='jd')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            lon = self.location.lon.deg
            lat = self.location.lat.deg
            alt = self.location.height.value

            dts = ap_times.datetime

            twilights = numpy.array([self._get_twilight(dt, lon, lat, alt)
                                     for dt in dts])

            # Convert to JD as array to reduce overhead in astropy.time.Time
            twilights_jd = astropy.time.Time(twilights).jd

            midnight = numpy.mean(twilights_jd, axis=1)

            # We call get_mood_data without location since it increases the
            # runtime significantly and we don't need the extra precision.
            moonpos, moonphase = get_moon_data(astropy.time.Time(midnight, format='jd'))

        self.data = astropy.table.Table(
            data=[jds, twilights_jd[:, 0], twilights_jd[:, 1],
                  moonpos.ra.deg, moonpos.dec.deg, moonphase],
            names=['JD', 'evening_twilight', 'morning_twilight',
                   'moon_ra', 'moon_dec', 'moon_phase'])

    def __repr__(self):

        start = self.data['JD'][0]
        end = self.data['JD'][-1]

        return f'<Observing plan (observatory={self.observatory!r}, dates=({start}, {end}))>'

    def __getitem__(self, item):
        """Overrides getitem to access the astropy table directly."""

        return self.data.__getitem__(item)

    def write(self, path, overwrite=False):
        """Writes the observing plan to a file."""

        self.data.write(path, format='ascii.fixed_width', delimiter='|', overwrite=overwrite)

    @classmethod
    def read(cls, path, observatory):
        """Reads a plan file.

        Parameters
        ----------
        path : path-like
            Path to a file containing an observing plan file. The file must be
            in ascii, fixed-width format with ``|`` as separators
        observatory : str
            The observatory, either ``'APO'`` or ``'LCO'``.

        """

        return cls(
            astropy.table.Table.read(path,
                                     format='ascii.fixed_width',
                                     delimiter='|'),
            observatory, twilight_alt=None)

    def _get_twilight(self, datetime_today, lon, lat, alt):
        """Returns the dusk and dawn times associated with a given JD."""

        dusk = self._astral.dusk_utc(datetime_today, lat, lon,
                                     observer_elevation=alt,
                                     depression=self.twilight_alt)

        dawn = self._astral.dawn_utc(datetime_today + _delta_dt, lat, lon,
                                     observer_elevation=alt,
                                     depression=self.twilight_alt)

        return dusk, dawn
