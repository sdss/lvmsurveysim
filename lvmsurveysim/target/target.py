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
import warnings

from lvmsurveysim.ifu import IFU
from lvmsurveysim.utils import plot as lvm_plot
import lvmsurveysim.utils.spherical
from lvmsurveysim.exceptions import LVMSurveyOpsError, LVMSurveyOpsWarning

from .. import config
from ..telescope import Telescope
from .skyregion import SkyRegion
from .tile import Tile


__all__ = ['Target', 'TargetList']


seaborn.set()


class Target(object):
    """A `.Region` with additional observing information.

    A `.Target` object is similar to a `.SkyRegion` but it is named and contains
    information about what telescope will observe it, its observing
    priority relative to all other targets, and a set of observing constraints
    ans strategies to be implemented during scheduling of the target (airmass,
    lunation, shadow height, tile order, ...).
    
    There is a special kind of target not represented internally as 
    a `.SkyRegion`, the fullsky target, which represents a (sparse) grid of tiles
    on the whole sky. 
    
    The Target constructor accepts the following keyword parameters, which are
    also available as keywords from a list read by `.from_list`. Typically a Target 
    will not be instatiated indivisually. The typical use case will involve the `.TargetList` 
    class which is initialize via a yaml configuration file, the survey 'target list'.

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
    min_shadowheight : float
        Minimum shadow height in km to observe the given target
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
    overlap:
        calculate overlap between this target and others and discard, defaults to true
    tile_union:
        tile_union that the target belongs to, if any; that is an area of sky that is tiled
        from a single hexagon grid to ensure gapless tiling of overlapping regions.
    geodesic:
        geodesic tiling of the full sphere instead of region
    sparse:
        sparse tiling factor, or depth value (number of subdivisions) in case of geodesic tiling
    group:
        (list of) group names the target belongs to (e.g. MilkyWay). used for aggregating survey statistics
        and plotting survey progress.

    Attributes
    ----------
    region : `.SkyRegion`
        The `.SkyRegion` object associated with this target.

    """

    def __init__(self, *args, **kwargs):

        self.name = kwargs.pop('name', '')
        self.priority = kwargs.pop('priority', 1)

        self.observatory = kwargs.pop('observatory', 'BOTH')
        self.max_airmass = kwargs.pop('max_airmass', 1.75)
        self.min_shadowheight = kwargs.pop('min_shadowheight', 1000.0)
        self.exptime = kwargs.pop('exptime', 900)
        self.n_exposures = kwargs.pop('n_exposures', 9)
        self.min_exposures = kwargs.pop('min_exposures', 3)
        self.min_moon_dist = kwargs.pop('min_moon_dist', 90)
        self.max_lunation = kwargs.pop('max_lunation', 1.0)
        self.overhead = kwargs.pop('overhead', 1.0)
        self.groups = kwargs.pop('group', [])
        self.tiling_strategy = kwargs.pop('tiling_strategy', 'lowest_airmass')
        self.tile_union = kwargs.pop('tile_union', None)
        self.overlap = kwargs.pop('overlap', True)
        self.geodesic = kwargs.pop('geodesic', False) # full sky tiling, use sparse for depth
        self.sparse = kwargs.pop('sparse', None)

        telescope = kwargs.pop('telescope', None)
        assert telescope is not None, 'must specify a telescope keyword.'

        if isinstance(telescope, Telescope):
            self.telescope = Telescope
        else:
            self.telescope = Telescope.from_config(telescope)

        self.region = SkyRegion(*args, **kwargs)

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
                min_shadowheight: 1000.0
                exptime: 900
                n_exposures: 1
                min_exposures: 1
                ...

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
            #               f'Using default IFU {ifu.name!r}.', LVMSurveyOpsWarning)

        assert pixarea is not None or (ifu is not None and telescope is not None), \
            'either pixarea or ifu and telescope need to be defined.'

        if pixarea is None:
            pixarea = (ifu.fibre_size / 2. * telescope.plate_scale).to('degree')**2 * numpy.pi
            pixarea *= ifu.n_fibres
            pixarea = pixarea.value

        return pixarea

    def get_tiling(self, ifu=None, telescope=None, to_frame=None):
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

        Returns
        -------
        tiles : list of `~lvmsurveysi.target.tile`
            A list of `~lvmsurveysi.target.tile` with the list of
            tile coordinates, priorities, and other data

        """

        telescope = telescope or self.telescope

        if ifu is None:
            ifu = IFU.from_config()
            # warnings.warn(f'target {self.name}: no IFU provided. '
            #               f'Using default IFU {ifu.name!r}.', LVMSurveyOpsWarning)

        print('Tiling target ' + self.name)
        coords = ifu.get_tile_grid(self.region, telescope.plate_scale, sparse=self.sparse, geodesic=self.geodesic)
        tiles = astropy.coordinates.SkyCoord(coords[:, 0], coords[:, 1], frame=self.frame, unit='deg')
        # second set offset in dec to find position angle after transform
        tiles2 = astropy.coordinates.SkyCoord(coords[:, 0], coords[:, 1]+1./3600, frame=self.frame, unit='deg')

        # transform not only centers, but also second set of coordinates slightly north, then compute the angle
        if to_frame:
            tiles = tiles.transform_to(to_frame)
            tiles2 = tiles2.transform_to(to_frame)
        self.pa = tiles.position_angle(tiles2)

        # cache the new tiles and the priorities
        self.tiles = tiles
        self.tile_priorities = self.get_tile_priorities()
        return self.make_tiles()


    def make_tiles(self):
        """ Return a list of `~lvmsurveysim.schedule.Tile` tile objects for this target.
        Requires the self.tiles, self.pa and self.tile_priorites arrays to have been 
        calculated using the `.get_tiling` method.

        """
        return [Tile(self.tiles[i], self.pa[i], self.tile_priorities[i]) for i in range(len(self.tiles))]

    def get_tiles_from_union(self, coords, pa):
        """ Select tiles belonging to this target from a list of coordinates.

        This method is used to select the tiles belonging to this target from a 
        list of coordinates of a tile union.

        Parameters
        ----------
        coords, pa : ~numpy.array
            vectors of coordinates and PAs of the tile union before selection.

        Returns
        -------
        coords, pa : ~numpy.array
            vectors of coordinates and PAs remaining in tile union after selection.

        """
        mask = numpy.full(len(coords), True)
        icrs_r = self.region.icrs_region()
        for i, c in enumerate(coords):
            if icrs_r.contains_point(c.ra.deg, c.dec.deg):
                mask[i] = False

        self.tiles = coords[~mask]
        self.pa = pa[~mask]
        self.tile_priorities = self.get_tile_priorities()
        return coords[mask], pa[mask]



    def get_tile_priorities(self):
        """Return an array with tile priorities according to the tiling
        strategy defined for this target.

        Returns
        -------
        priorities: ~numpy.array
            Array of length of number of tiles with the priority for each tile.

        """
        if len(self.tiles) == 0:
            warnings.warn(f'target {self.name}: no tiles when calling get_tile_priorities(). ', LVMSurveyOpsWarning)
            return numpy.array([])

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


    def get_skyregion(self):
        """ Return the `.SkyRegion` of the target
        """
        return self.region

    def is_sparse(self):
        if self.sparse == None:
            return False
        else:
            return True

    def density(self):
        if self.is_sparse():
            return 1.0/self.sparse
        else:
            return 1.0

    def in_tile_union_with(self, other):
        return (self.tile_union != None) and (self.tile_union==other.tile_union)


    def plot(self, *args, **kwargs):
        """Plots the region. An alias for ``.SkyRegion.plot``.
        
        """

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

        c1,c2 = lvm_plot.convert_to_mollweide(lon, lat)

        ax.scatter(c1, c2, **kwargs)

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
                    os.path.expanduser(os.path.expandvars(config['tiledb']['target_file'])))
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

    def get_groups(self):
        """Returns a list of all the groups for all the targets in the list."""

        groups = set()
        for target in self:
            groups.update(target.groups)

        return list(groups)

    def get_tile_unions(self):
        """Returns a list of all the tile unions in the target list."""

        unions = set()
        for target in self:
            if target.tile_union:
                unions.update([target.tile_union])

        return list(unions)

    def get_union_targets(self, tile_union):
        """Returns the targets that are in a tile union.

        Parameters
        ----------
        tile_union : str
            The group name.

        Returns
        -------
        targets : `list`
            A list of target names that are included in ``tile_union``.

        """

        ut = []

        for target in self:
            if tile_union == target.tile_union:
                ut.append(target)

        return TargetList(targets=ut)

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
            (i.e., a `~lvmsurveysim.target.Tile` list with the
            coordinates, priorities and other data of the tiles).

        """
        return {ii: self[ii].get_tiling(**kwargs) for ii in range(len(self))}


    def order_by_priority(self):
        """ Return a copy of the target list ordered by priorities highest to lowest.

        """
        def prio(t):
            return t.priority
        
        return sorted(self, key=prio, reverse=True)


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
