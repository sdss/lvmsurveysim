#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 27, 2017
# @Filename: telescope.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import astropy.units as uu

import lvmsurveysim


class Telescope(object):
    """A class to represent a telescope.

    Holds information about a specific telescope such as diameter and focal
    ratio. Calculates derived quantities such as focal length and plate scale.
    A `.Telescope` object can be instantiated with a name followed by a
    series of values, or just with the name. In the latter case the necessary
    information must be contained in the configuration file, under the
    ``telescopes`` field.

    Parameters:
        name (str):
            The name of the telescope. If only the ``name`` is provided, the
            information will be grabbed from the configuration file.
        diameter (float):
            The diameter of the telescope, in meters
        f (float):
            The f number of the telescope.

    """

    def __init__(self, name, diameter=None, f=None):

        self.name = name

        if f is None and diameter is None:
            assert 'telescopes' in lvmsurveysim.config, \
                'configuration does not have telescopes section.'
            assert self.name in lvmsurveysim.config['telescopes'], \
                f'telescope name {self.name!r} not found in configuration.'

            self.diameter = lvmsurveysim.config['telescopes'][self.name]['diameter'] * uu.meter
            self.f = lvmsurveysim.config['telescopes'][self.name]['f']

        else:
            assert all([diameter, f]), 'both diameter and f must be defined.'
            self.diameter = diameter * uu.meter
            self.f = f

    def __repr__(self):

        return f'<Telescope {self.name!r}>'

    @property
    def focal_length(self):
        """Returns the focal length of the telescope."""

        return self.f * self.diameter

    @property
    def plate_scale(self):
        """Returns the plate scale as an `~astropy.units.Quantity`."""

        return 206265 * uu.arcsec / (self.diameter.to('mm') * self.f)
