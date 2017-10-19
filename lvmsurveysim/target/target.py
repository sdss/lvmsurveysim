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

import astropy.coordinates

from .regions import EllipticalRegion
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
            target.
        region_mode ({'ellipse', 'rectangle', 'polygon'}):
            The type of area that ``region_params`` specify.
        region_params:
            A list of parameters that define the area on the sky of the
            target. See :ref:`target-defining`.

    """

    def __init__(self, name, coords, region_mode=None, region_params=None):

        self.coords = coords

        if not isinstance(coords, astropy.coordinates.SkyCoord):
            assert len(coords) == 2, 'invalid number of coordinates.'
            self.coords = astropy.coordinates.SkyCoord(ra=coords[0], dec=coords[1], unit='deg')

        if region_mode is not None:
            assert region_params is not None, \
                'region_mode and region_params must be both None or not None.'
            self.region = self._create_area(coords, region_mode, region_params)

    @staticmethod
    def _create_area(coords, region_mode, region_params):
        """Returns a `Shapely`_ object with the region on the sky."""

        if region_mode == 'ellipse':
            region = EllipticalRegion(coords, region_params[0] / 2.,
                                      b=region_params[1] / 2.,
                                      pa=region_params[2])
        else:
            raise ValueError(f'invalid region_mode={region_mode!r}.')

        return region

    @classmethod
    def from_target_list(cls, name, target_list=None):
        """Returns an instance of `.Target` from a target list.

        Initialises a new target whose parameters have been previously defined
        in a target list. Target lists must be YAML files in which each
        target has attributes ``coords``, ``region_mode``, and
        ``region_params``, defined as in :ref:`target-defining` For example:

        .. code-block:: yaml

            M81:
                coords: [148.888333, 69.0652778]
                region_mode: 'ellipse'
                region_params: [0.209722, 0.106958333, 149]

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

        return cls(name, target['coords'], region_mode=target['region_mode'],
                   region_params=target['region_params'])
