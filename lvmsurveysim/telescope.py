#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2017-10-27
# @Filename: telescope.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-02-21 18:57:17

import astropy.units

from lvmsurveysim import config


__all__ = ['Telescope']


class Telescope(object):
    """A class to represent a telescope.

    Holds information about a specific telescope such as diameter and focal
    ratio. Calculates derived quantities such as focal length and plate scale.
    A `.Telescope` object can be instantiated with a name followed by a
    series of values, or just with the name. In the latter case the necessary
    information must be contained in the configuration file, under the
    ``telescopes`` field.

    Parameters
    ----------
    name : str
        The name of the telescope. If only the ``name`` is provided, the
        information will be grabbed from the configuration file.
    diameter : float
        The diameter of the telescope, in meters
    f : float
        The f number of the telescope.

    """

    def __init__(self, name, diameter=None, f=None):

        self.name = name

        assert diameter is not None and f is not None, 'both diameter and f must be defined.'

        self.diameter = diameter * astropy.units.meter
        self.f = f

    def __repr__(self):

        return f'<Telescope {self.name!r}>'

    @classmethod
    def from_config(cls, name):
        """Loads a telescope configuration from the configuration file.

        Example
        -------

        >>> lvm160 = Telescope.from_config('LVM-160)

        """

        assert 'telescopes' in config, 'configuration does not have telescopes section.'
        assert name in config['telescopes'], f'telescope name {name!r} not found in configuration.'

        return cls(name, diameter=config['telescopes'][name]['diameter'],
                   f=config['telescopes'][name]['f'])

    @property
    def focal_length(self):
        """Returns the focal length of the telescope."""

        return self.f * self.diameter

    @property
    def plate_scale(self):
        """Returns the plate scale as an `~astropy.units.Quantity`."""

        return 206265 * astropy.units.arcsec / (self.diameter.to('mm') * self.f)
