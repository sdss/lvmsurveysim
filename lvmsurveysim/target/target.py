#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 10, 2017
# @Filename: target.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pathlib
import yaml

from . import regions
from .. import config


class Target(object):
    """A representation of an astronomical target.

    Defines a target, including target centre, area on the sky, surface
    brightnes, etc. See the section :ref:`target-defining` for more
    information.

    Parameters:
        name (str):
            The identifier of this target, e.g., ``'M81'``.
        coords (tuple or `~astropy.coordinates.SkyCoord`):
            A tuple of ``(ra, dec)`` in degrees or a
            `~astropy.coordinates.SkyCoord` describing the centre of the
            target. If the region is of type ``polygon``, ``coords`` must
            be a list of vertices as indicated in `~.regions.PolygonalRegion`.
        region_type (str):
            One of the valid region types for `~.regions.Region`.
        region_params (dict):
            A dictionary of parameters to be passed to `~.regions.Region`.

    Example:

        >>> target = Target('MyTarget', coords=(169, 65), region_type='circle', region_params={'r': 0.1})
        >>> target
        <Region 'MyTarget'>
        >>> target.region
        <CircularRegion (coords=<SkyCoord (ICRS): (ra, dec) in deg
              ( 169.,  65.)>, r=0.100 deg)>

    """

    def __init__(self, name, coords, region_type, region_params={}):

        self.name = name
        self.coords = coords
        self.region = self._create_region(coords, region_type, region_params)

    def __repr__(self):

        return f'<Region {self.name!r}>'

    @staticmethod
    def _create_region(coords, region_type, region_params):
        """Returns a `.regions.Region` with the target on the sky."""

        return regions.Region(region_type, coords, **region_params)

    @classmethod
    def from_target_list(cls, name, target_list=None):
        """Returns an instance of `.Target` from a target list.

        Initialises a new target whose parameters have been previously defined
        in a target list. Target lists must be YAML files in which each
        target has attributes ``coords``, ``region_params``, and
        ``region_params``, defined as in :ref:`target-defining`. For example:

        .. code-block:: yaml

            M81:
                coords: [148.888333, 69.0652778]
                region_type: 'ellipse'
                region_params:
                    a: 0.209722
                    b: 0.106958333
                    pa: 149

        Parameters:
            name (str):
                The identifier for the target. Must be defined in the target
                list file.
            target_list (str, `~pathlib.Path`, or None):
                The path to the YAML file containing the target list. If
                ``None``, default to the target list contained in ``lvmcore``.

        Example:
            >>> from lvmsurveysim.target import Target
            >>> m81 = Target.from_target_list('M81')

        """

        if target_list is None:
            target_list = pathlib.Path(
                os.path.expanduser(os.path.expandvars(config['target_list'])))
        else:
            target_list = pathlib.Path(target_list)

        assert target_list.exists()

        targets = yaml.load(open(str(target_list)))

        assert name in targets, 'target not found in target list.'

        target = targets[name]

        return cls(name, target['coords'], region_type=target['region_type'],
                   region_params=target['region_params'])
