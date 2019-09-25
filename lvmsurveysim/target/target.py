#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-02-19
# @Filename: target.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-09-25 15:20:31

import os
import pathlib

import astropy
import numpy
import seaborn
import yaml

from lvmsurveysim.ifu import IFU
from lvmsurveysim.utils import plot as lvm_plot
import lvmsurveysim.utils.spherical

from .. import config
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
    max_airmass : float
        Maximum air mass to observe the given target
    exptime : float
        Exposure time of an individual pointing
    n_exposures
        Number of individual pointings to reach desired S/N
    min_exposures : int
        Minimum number of exposures to make a "good visit"
    min_moon_dist : float
        Minimum moon distance between target before observations are
        called off.
    max_lunation : float
        The maximum lunation (fraction of moon illuminated,
        number between 0 and 1)
    overhead : float
        The overhead factor per exposure quantum for this target's observing
        scheme.

    Attributes
    ----------
    region : `.Region`
        The `.Region` object associated with this target.

    """

    def __init__(self, *args, **kwargs):

        self.name = kwargs.pop('name', '')
        self.priority = kwargs.pop('priority', 1)

        self.observatory = kwargs.pop('observatory', 'BOTH')
        self.max_airmass = kwargs.pop('max_airmass', 1.75)
        self.exptime = kwargs.pop('exptime', 900)
        self.n_exposures = kwargs.pop('n_exposures', 9)
        self.min_exposures = kwargs.pop('min_exposures', 3)
        self.min_moon_dist = kwargs.pop('min_moon_dist', 90)
        self.max_lunation = kwargs.pop('max_lunation', 1.0)
        self.overhead = kwargs.pop('overhead', 1.0)
        self.groups = kwargs.pop('group', [])
        self.tiling_strategy = kwargs.pop('tiling_strategy', 'lowest_airmass')

        telescope = kwargs.pop('telescope', None)
        assert telescope is not None, 'must specify a telescope keyword.'

        if isinstance(telescope, Telescope):
            self.telescope = Telescope
        else:
            self.telescope = Telescope.from_config(telescope)

        self.region = Region(*args, **kwargs)

        self.frame = self.region.frame

        self.tiles = None
        self.tile_priorities = None

    def __repr__(self):

        return (f'<Target (name={self.name!r}, telescope={self.telescope.name!r}, '
                f'region_type={self.region.region_type!r})>')

    @classmethod
    def from_list(cls, name, targets=None):
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
                observatory: APO {LCO, BOTH}
                telecope: LVM-1m {LVM-160}
                max_airmass: 1.75
                exptime: 900
                n_exposures: 1
                min_exposures: 1


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

        assert targets is not None, "target dictionary not defined"

        assert name in targets, 'target not found in target list.'

        target = targets[name]

        region_type = target.pop('region_type')
        coords = target.pop('coords')
        region_params = target.pop('region_params', {})

        target.update(region_params)

        return cls(region_type, coords, name=name, **target)

    def get_pixarea(self, pixarea=None, ifu=None, telescope=None):
        """Gets the size of the tile in square degrees."""

        telescope = telescope or self.telescope

        if ifu is None:
            ifu = IFU.from_config()
            # warnings.warn(f'target {self.name}: no IFU provided. '
            #               f'Using default IFU {ifu.name!r}.', LVMSurveySimWarning)

        assert pixarea is not None or (ifu is not None and telescope is not None), \
            'either pixarea or ifu and telescope need to be defined.'

        if pixarea is None:
            pixarea = (ifu.fibre_size / 2. * telescope.plate_scale).to('degree')**2 * numpy.pi
            pixarea *= ifu.n_fibres
            pixarea = pixarea.value

        return pixarea

    def get_tiling(self, ifu=None, telescope=None, to_frame=None, force_retile=False):
        """Tessellates the target region and returns a list of tile centres.

        Parameters
        ----------
        ifu : ~lvmsurveysim.tiling.IFU
            The IFU used for tiling the region. If not provided, the default
            one is used.
        telescope : ~lvmsurveysim.telescope.Telescope
            The telescope on which the IFU is mounted. Defaults to the object
            ``telescope`` attribute.
        to_frame : str
            If ``return_coords``, the reference frame in which the coordinates
            should be returned. If `None`, defaults to the region internal
            reference frame.
        force_retile : bool
            Force recalculation of the tiles. Tiles are cached once calculated
            and the cached ones are returned unless this flag is set.

        Returns
        -------
        pixels : `~astropy.coordinates.SkyCoord`
            A list of `~astropy.coordinates.SkyCoord` with the list of
            tile centre coordinates.

        """

        # return cached values unless told not to
        if self.tiles is not None:
            if force_retile is False:
                return self.tiles

        telescope = telescope or self.telescope

        if ifu is None:
            ifu = IFU.from_config()
            # warnings.warn(f'target {self.name}: no IFU provided. '
            #               f'Using default IFU {ifu.name!r}.', LVMSurveySimWarning)

        tiles = ifu.get_tile_grid(self.region, telescope.plate_scale)
        tiles = astropy.coordinates.SkyCoord(tiles[:, 0], tiles[:, 1],
                                             frame=self.frame, unit='deg')

        if to_frame:
            tiles = tiles.transform_to(to_frame)

        # cache the new tiles and invalidate the priorities
        self.tiles = tiles
        self.tile_priorities = None

        return tiles

    def get_tile_priorities(self, force_retile=False):
        """Return an array with tile priorities according to the tiling
        strategy defined for this target.

        Returns
        -------
        priorities: ~numpy.array
            Array of length of number of tiles with the priority for each tile.
        """

        # return cached values unless told not to
        if self.tile_priorities is not None:
            if force_retile is False:
                return self.tile_priorities

        if self.tiling_strategy == 'lowest_airmass':
            self.tile_priorities = numpy.ones(len(self.tiles), dtype=int)
        elif self.tiling_strategy == 'center_first':
            self.tile_priorities = self.center_first_priorities_()
        else:
            raise ValueError(f'invalid tiling strategy: {self.tiling_strategy}.')

        return self.tile_priorities

    def center_first_priorities_(self):
        """Return an array with tile priorities according for the center-first
        tiling strategy.

        Tiles are prioritized according to the distance from the region
        barycenter. Priorities are equal along lines of constant distance
        from the barycenter, quantized in units of the tile diameter.

        Returns
        -------
        priorities : ~numpy.array
            Array of length of number of tiles with the priority for each tile.

        """
        r, d = self.tiles.ra.deg, self.tiles.dec.deg

        # TODO: proper calculation of barycenter on the sphere!
        rc = numpy.average(r)
        dc = numpy.average(d)
        dist = lvmsurveysim.utils.spherical.great_circle_distance(r, d, rc, dc)
        field = numpy.sqrt(self.get_pixarea() / numpy.pi)  # TODO: better way to get field size!!!

        p = numpy.floor(dist / field).astype(int)
        return numpy.max(p) - p + 1  # invert since priorities increase with value

    def plot(self, *args, **kwargs):
        """Plots the region. An alias for ``.Region.plot``."""

        return self.region.plot(*args, **kwargs)

    def plot_tiling(self, coords=None, ifu=None, frame=None, fig=None, **kwargs):
        """Plots the tiles within the region.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            A list of `~astropy.coordinates.SkyCoord` to plot. If not provided,
            `~.Target.get_tiling` will be called with the options below.
        ifu : ~lvmsurveysim.tiling.IFU
            The IFU used for tiling the region. If not provided, the default
            one is used.
        frame : str
            The reference frame on which the pixels will be displayed. Defaults
            to the internal frame of the target.
        ax : ~matplotlib.axes.Axes
            A Matplotlib `~matplotlib.axes.Axes` object to use. Otherwise, a
            new one will be created.
        kwargs : dict
            Parameters to be passed to `~matplotlib.axes.scatter`.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            The Matplotlib `~matplotlib.figure.Figure`.

        """

        frame = frame or self.frame

        if coords is None:
            coords = self.get_tiling(ifu=ifu, to_frame=frame)

        if frame == 'icrs':
            lon, lat = coords.ra.deg, coords.dec.deg
        elif frame == 'galactic':
            lon, lat = coords.l.deg, coords.b.deg

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

        self.filename = None

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

            self.filename = target_file

            targets_dict = yaml.load(open(str(target_file)), Loader=yaml.FullLoader)

            self._names = list(targets_dict.keys())

            targets = [Target.from_list(name, targets=targets_dict)
                       for name in self._names]

            super().__init__(targets)

    def get_target(self, name):
        """Returns the target whose name correspond to ``name``."""

        return self[self._names.index(name)]

    def get_group_targets(self, group, primary=True):
        """Returns the targets that are in a group.

        Parameters
        ----------
        group : str
            The group name.
        primary : bool
            Return only the target if ``group`` is the primary group to which
            the target belongs (i.e., the first one in the list).

        Returns
        -------
        targets : `list`
            A list of target names that are included in ``group``.

        """

        targets = []

        for target in self:
            if group in target.groups:
                if (primary and group == target.groups[0]) or (not primary):
                    targets.append(target.name)

        return targets

    def list_groups(self):
        """Returns a list of all the groups for all the targets in the list."""

        groups = []
        for target in self:
            groups += target.groups

        return list(set(groups))

    def get_tiling(self, **kwargs):
        """Gets the tile centres for all the targets in the set.

        Parameters
        ----------
        kwargs : dict
            Parameters to be passed to `.Target.get_tiling`.

        Returns
        -------
        tiling : dict
            A dictionary in which the key is the index of the target in the
            `.TargetList` and its value the output of
            `.Target.get_tiling` called with ``kwarg`` parameters
            (i.e., a `~astropy.coordinates.SkyCoord` object with the
            position of the tile centres).

        """

        return {ii: self[ii].get_tiling(**kwargs) for ii in range(len(self))}

    def get_tile_priorities(self, **kwargs):
        """Gets the tile priorities for all the targets in the set.

        Parameters
        ----------
        kwargs : dict
            Parameters to be passed to `.Target.get_tile_priorities`.

        Returns
        -------
        tiling : dict
            A dictionary in which the key is the index of the target in the
            `.TargetList` and its value the output of
            `.Target.get_tile_priorities` called with ``kwarg`` parameters.

        """

        return {ii: self[ii].get_tile_priorities(**kwargs) for ii in range(len(self))}

    def plot_tiling(self, frame='icrs', **kwargs):
        """Plots all the target pixels in a single Mollweide projection.

        Parameters
        ----------
        frame : str
            The coordinate frame to which all the pixel centres will be
            converted.
        kwargs : dict
            Parameters to be passed to `.Target.plot_tiling`. By default, each
            target will be plotted in a different colour.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            The Matplotlib `~matplotlib.figure.Figure`.

        """

        assert len(self) > 0, 'no targets in list.'

        zorder = 100

        fig = self[0].plot_tiling(frame=frame, zorder=zorder, **kwargs)

        if len(self) > 1:
            for target in self[1:]:
                zorder -= 1
                fig = target.plot_tiling(fig=fig, frame=frame, zorder=zorder, **kwargs)

        return fig
