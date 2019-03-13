#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-02-19
# @Filename: target.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-13 13:12:04

import os
import pathlib
import warnings

import astropy
import healpy
import matplotlib.pyplot
import numpy
import seaborn
import yaml

import lvmsurveysim.utils.healpix
from lvmsurveysim.utils import plot as lvm_plot

from . import _VALID_FRAMES
from .. import config
from ..exceptions import LVMSurveySimWarning
from ..ifu import IFU
from ..telescope import Telescope
from .region import Region


__all__ = ['Target', 'TargetList']


seaborn.set()


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

        self.frame = self.region.frame

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

        targets = yaml.load(open(str(target_file)), Loader=yaml.FullLoader)

        assert name in targets, 'target not found in target list.'

        target = targets[name]

        region_type = target.pop('region_type')
        coords = target.pop('coords')
        region_params = target.pop('region_params', {})

        target.update(region_params)

        return cls(region_type, coords, name=name, **target)

    def _get_nside(self, pixarea=None, ifu=None, telescope=None):
        """Gets the ``nside`` for a pixarea or IFU/telescope configuration."""

        telescope = telescope or self.telescope

        if ifu is None:
            ifu = IFU.from_config()
            warnings.warn(f'target {self.name}: no IFU provided. '
                          f'Using default IFU {ifu.name!r}.', LVMSurveySimWarning)

        assert pixarea is not None or (ifu is not None and telescope is not None), \
            'either pixarea or ifu and telescope need to be defined.'

        if pixarea is None:
            pixarea = (ifu.fibre_size / 2. * telescope.plate_scale).to('degree')**2 * numpy.pi
            pixarea *= ifu.n_fibres
            pixarea = pixarea.value

        return lvmsurveysim.utils.healpix.get_minimum_nside_pixarea(pixarea)

    def get_healpix_tiling(self, pixarea=None, ifu=None, telescope=None,
                           return_coords=False, to_frame=None, inclusive=True):
        """Tessellates the target region and returns a list of HealPix pixels.

        Parameters
        ----------
        pixarea : float
            Desired area of the HealPix pixel, in square degrees. The HealPix
            order that produces a pixel of size equal or smaller than
            ``pixarea`` will be used.
        ifu : ~lvmsurveysim.tiling.IFU
            The IFU used for tiling the region. If not provided, the default
            one is used.
        telescope : ~lvmsurveysim.telescope.Telescope
            The telescope on which the IFU is mounted. Defaults to the object
            ``telescope`` attribute.
        return_coords : bool
            If `True`, returns the coordinates of the included pixels instead
            of their value.
        to_frame : str
            If ``return_coords``, the reference frame in which the coordinates
            should be returned. If `None`, defaults to the region internal
            reference frame.
        inclusive : bool
            Whether to return pixels that only partially overlap with the
            target region. Note that this is only approximated (see the
            explanation in `~lvmsurveysim.utils.healpix.tile_geometry`).


        Returns
        -------
        pixels : `numpy.ndarray` or `~astropy.coordinates.SkyCoord`
            A list of HealPix pixels that tile this target. Only pixels whose
            centre is contained in the region are included. If
            ``return_coords=True``, returns a `~astropy.coordinates.SkyCoord`
            with the list of coordinates.

        """

        if to_frame is not None:
            assert to_frame in _VALID_FRAMES, 'invalid frame'

        nside = self._get_nside(pixarea=pixarea, ifu=ifu, telescope=telescope)

        pixels = lvmsurveysim.utils.healpix.tile_geometry(self.region.shapely, nside,
                                                          return_coords=return_coords,
                                                          inclusive=inclusive)

        if return_coords:
            coords = astropy.coordinates.SkyCoord(pixels[:, 0], pixels[:, 1],
                                                  frame=self.frame, unit='deg')
            if to_frame is not None:
                coords = coords.transform_to(to_frame)
            return coords

        return pixels

    def plot(self, *args, **kwargs):
        """Plots the region. An alias for ``.Region.plot``."""

        return self.region.plot(*args, **kwargs)

    def plot_healpix(self, coords=None, ifu=None, frame=None, fig=None,
                     use_healpy=False, **kwargs):
        """Plots the region as HealPix pixels.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            A list of `~astropy.coordinates.SkyCoord` to plot. If not provided,
            `~.Target.get_healpix` will be called with the options below.
        ifu : ~lvmsurveysim.tiling.IFU
            The IFU used for tiling the region. If not provided, the default
            one is used.
        frame : str
            The reference frame on which the pixels will be displayed. Defaults
            to the internal frame of the target.
        ax : ~matplotlib.axes.Axes
            A Matplotlib `~matplotlib.axes.Axes` object to use. Otherwise, a
            new one will be created.
        use_healpy : bool
            If `True`, uses the Healpy Mollweide plotting system.
        kwargs : dict
            Parameters to be passed to `~matplotlib.axes.scatter` (or
            `~healpy.visufunc.mollview` if ``use_healpix=True``).

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            The Matplotlib `~matplotlib.figure.Figure`.

        """

        frame = frame or self.frame

        if coords is None:
            coords = self.get_healpix_tiling(ifu=ifu, return_coords=True,
                                             to_frame=frame)

        if frame == 'icrs':
            lon, lat = coords.ra.deg, coords.dec.deg
        elif frame == 'galactic':
            lon, lat = coords.l.deg, coords.b.deg

        if use_healpy:

            if 'fig' not in kwargs:
                kwargs['fig'] = matplotlib.pyplot.Figure()

            # Convert coordinates to pixels.
            nside = self._get_nside(ifu=ifu)
            pixels = healpy.ang2pix(nside, lon, lat, nest=True, lonlat=True)

            npix = numpy.zeros(healpy.nside2npix(nside)) + kwargs.pop('value', 1)
            mask = numpy.ones(healpy.nside2npix(nside), dtype=numpy.bool)
            mask[pixels] = False

            healmap = healpy.ma(npix)
            healmap.mask = mask

            healpy.mollview(healmap.filled(), nest=True, cbar=False, hold=True, **kwargs)
            healpy.graticule(dpar=30, dmer=30)

            return kwargs['fig']

        if fig is None:
            fig, ax = lvm_plot.get_axes(projection='mollweide', frame=frame)
        else:
            ax = fig.axes[0]

        coords_array = numpy.array([lon, lat]).T
        coords_moll = lvm_plot.convert_to_mollweide(coords_array)

        ax.scatter(coords_moll[:, 0], coords_moll[:, 1], **kwargs)

        return fig


class TargetList(list):
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

    def __init__(self, targets=None, target_file=None):

        if targets:

            self._names = [target.name for target in targets]
            super().__init__(targets)

        else:

            if target_file is None:
                target_file = pathlib.Path(
                    os.path.expanduser(os.path.expandvars(config['target_file'])))
            else:
                target_file = pathlib.Path(target_file)

            assert target_file.exists()

            targets_dict = yaml.load(open(str(target_file)), Loader=yaml.FullLoader)

            self._names = list(targets_dict.keys())

            targets = [Target.from_list(name, target_file=target_file)
                       for name in self._names]

            super().__init__(targets)

    def get_target(self, name):
        """Returns the target whose name correspond to ``name``."""

        return self[self._names.index(name)]

    def get_healpix_tiling(self, **kwargs):
        """Gets the HealPix coverage for all the targets in the set.

        Parameters
        ----------
        kwargs : dict
            Parameters to be passed to `.Target.get_healpix_tiling`.

        Returns
        -------
        tiling : dict
            A dictionary in which the key is the index of the target in the
            `.TargetList` and its value the output of
            `.Target.get_healpix_tiling` called with ``kwarg`` parameters
            (i.e., either an array of HealPix pixels at ``nside`` resolution
            or a `~astropy.coordinates.SkyCoord` object with the position of
            the pixel centres).

        """

        return {ii: self[ii].get_healpix_tiling(**kwargs) for ii in range(len(self))}

    def plot_healpix(self, frame='icrs', **kwargs):
        """Plots all the target pixels in a single Mollweide projection.

        Parameters
        ----------
        frame : str
            The coordinate frame to which all the pixel centres will be
            converted.
        kwargs : dict
            Parameters to be passed to `.Target.plot_healpix`. By default, each
            target will be plotted on a different colour.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            The Matplotlib `~matplotlib.figure.Figure`.

        """

        assert len(self) > 0, 'no targets in list.'

        zorder = 100

        fig = self[0].plot_healpix(frame=frame, zorder=zorder, **kwargs)

        if len(self) > 1:
            for target in self[1:]:
                zorder -= 1
                fig = target.plot_healpix(fig=fig, frame=frame, zorder=zorder, **kwargs)

        return fig
