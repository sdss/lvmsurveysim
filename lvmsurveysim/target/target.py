#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-02-19
# @Filename: target.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-02-22 16:25:47

import os
import pathlib

import numpy
import yaml

from .. import config, log
from ..ifu import IFU
from ..telescope import Telescope
from .region import Region


__all__ = ['Target', 'TargetSet']


class Target(object):
    """A `.Region` with additional observing information.

    A `.Target` object is similar to a `.Region` but it is named and contains
    information about what telescope will observe it and its observing
    priority. It is instantiated as a `.Region` but accepts the following extra
    parameters.

    Parameters
    ----------
    name : str
        The name of the target.
    priority : int
        The priority at which this target should be observed. Higher numbers
        mean higher priority.
    telescope : str
        The telescope that will observe the target. Must be a string that
        matches a telescope entry in the configuration file or a
        `~lvmsurveysim.telescope.Telescope` instance.

    Attributes
    ----------
    region : `.Region`
        The `.Region` object associated with this target.

    """

    def __init__(self, *args, **kwargs):

        self.name = kwargs.pop('name', '')
        self.priority = kwargs.pop('priority', 1)

        telescope = kwargs.pop('telescope', None)
        assert telescope is not None, 'must specify a telescope keyword.'

        if isinstance(telescope, Telescope):
            self.telescope = Telescope
        else:
            self.telescope = Telescope.from_config(telescope)

        self.region = Region(*args, **kwargs)

    def __repr__(self):

        return (f'<Target (name={self.name!r}, telescope={self.telescope.name!r}, '
                f'region_type={self.region.region_type!r})>')

    @classmethod
    def from_list(cls, name, target_file=None):
        """Returns an instance of `.Target` from a target list.

        Initialises a new `.Target` whose parameters have been previously
        defined in a target list. Target lists must be YAML files in which each
        target defines region and the telescope that will observe it, as
        detailed in :ref:`target-defining`. For example:

        .. code-block:: yaml

            M81:
                coords: [148.888333, 69.0652778]
                region_type: ellipse
                frame: icrs
                region_params:
                    a: 0.209722
                    b: 0.106958333
                    pa: 149
                priority: 1
                telecope: APO 1-m

        Parameters
        ----------
        name : str
            The identifier for the target. Must be defined in the region.
            list file.
        target_file : `str`, `~pathlib.Path`, or `None`
            The path to the YAML file containing the region list. If
            `None`, default to the target list contained in ``lvmcore``.

        Example:
            >>> from lvmsurveysim.target import Target
            >>> m81 = Target.from_list('M81')

        """

        if target_file is None:
            target_file = pathlib.Path(
                os.path.expanduser(os.path.expandvars(config['target_file'])))
        else:
            target_file = pathlib.Path(target_file)

        assert target_file.exists()

        targets = yaml.load(open(str(target_file)))

        assert name in targets, 'target not found in target list.'

        target = targets[name]

        region_type = target.pop('region_type')
        coords = target.pop('coords')
        region_params = target.pop('region_params', {})

        target.update(region_params)

        return cls(region_type, coords, name=name, **target)

    def get_healpix(self, pixarea=None, ifu=None, telescope=None,
                    inclusive=True, return_coords=False):
        """Tessellates the target region and returns a list of HealPix pixels.

        Parameters
        ----------
        pixarea : float
            Desired area of the HealPix pixel, in square degrees. The HealPix
            order that produces a pixel of size equal or smaller than
            ``pixarea`` will be used.
        ifu : `~lvmsurveysim.tiling.IFU`
            The IFU used for tiling the region.
        telescope : `~lvmsurveysim.telescope.Telescope`
            The telescope on which the IFU is mounted. Defaults to the object
            ``telescope`` attribute.
        inclusive : bool
            Whather to include the HealPix pixels that overlap with the region
            but are not completely contained by it.
        return_coords : bool
            If True, returns the coordinates of the included pixels instead of
            their value.

        """

        import healpy

        telescope = telescope or self.telescope

        if ifu is None:
            ifu = IFU.from_config()
            log.warning(f'no IFU provided. Using default IFU {ifu.name!r}.')

        assert pixarea is not None or ifu is not None or telescope is not None, \
            'either pixarea or ifu and telescope need to be defined.'

        if pixarea is None:
            pixarea = (ifu.fibre_size / 2. * telescope.plate_scale).to('degree')**2 * numpy.pi
            pixarea *= ifu.n_fibres
            pixarea = pixarea.value

        order = 0
        while order <= 30:
            if healpy.pixelfunc.nside2pixarea(2**order, degrees=True) <= pixarea:
                break
            order += 1

        if order == 30:
            raise ValueError('pixarea is too small.')

        cartesian = numpy.array(self.region.to_cartesian()).T
        while numpy.all(cartesian[0] == cartesian[-1]):
            cartesian = numpy.delete(cartesian, [cartesian.shape[0] - 1], axis=0)

        pixels = healpy.query_polygon(2**order, cartesian, inclusive=inclusive)

        if return_coords:
            return numpy.array(healpy.pixelfunc.pix2ang(2**order, pixels, lonlat=True)).T
        return pixels

    def plot(self, *args, **kwargs):
        """Plots the region. An alias for ``.Region.plot``."""

        return self.region.plot(*args, **kwargs)


class TargetSet(list):
    """A list of all the targets to observe.

    Parameters
    ----------
    target_file : str
        The YAML file with all the targets to observe. Defaults to the
        ``lvmcore`` target list.

    Returns
    -------
    target_set : list
        A list of `.Target` instances.

    """

    def __init__(self, target_file=None):

        if target_file is None:
            target_file = pathlib.Path(
                os.path.expanduser(os.path.expandvars(config['target_file'])))
        else:
            target_file = pathlib.Path(target_file)

        assert target_file.exists()

        targets_dict = yaml.load(open(str(target_file)))

        names = targets_dict.keys()

        targets = [Target.from_list(name, target_file=target_file) for name in names]

        super().__init__(targets)
