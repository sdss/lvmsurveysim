#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 10, 2017
# @Filename: tiling.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-02-21 13:57:28


from __future__ import absolute_import, division, print_function

import copy

import astropy.coordinates
import astropy.units
import matplotlib
import numpy as np
import shapely.affinity
import shapely.geometry

from ..target import Region
from .ifu import IFU


__all__ = ['Tiling', 'Tile']


class Tile(object):
    """A tiling element.

    A `.Tile` is basically an `.IFU` with a position, scale, and rotation. A
    `.Tiling` is composed of multiple Tiles that optimally cover a region or
    target.

    Parameters:
        ifu (`.IFU`):
            The `.IFU` to use.
        coords (`~astropy.coordinates.SkyCoord` or tuple):
            The ``(RA, Dec)`` coordinates on which the centre of mass of the
            `.IFU` will be placed.
        scale (float):
            The plate scale, used to convert IFU physical units to on-sky
            positions and angles. In units of arcsec/mm.
        angle (`~astropy.coordinates.Angle` or float):
            The rotation angle of the `.IFU` measured from North to East.

    """

    def __init__(self, ifu, coords, scale, angle=0):

        assert isinstance(ifu, IFU), 'ifu is not a valid input type.'
        self.ifu = copy.deepcopy(ifu)

        if not isinstance(coords, astropy.coordinates.SkyCoord):
            coords = astropy.coordinates.SkyCoord(*coords, unit='deg')

        self.coords = coords
        self.angle = astropy.coordinates.Angle(angle, unit='deg')
        self._scale = scale

        self.transform(self._scale)

    def __repr__(self):

        return (f'<Tile RA={self.coords.ra.deg:.3f}, '
                f'Dec={self.coords.dec.deg:.3f}, '
                f'angle={self.angle.deg:.1f}>')

    def transform(self, scale=None):
        """Relocates/rescales the IFU shapely objects and gap centres."""

        if scale is not None:
            self._scale = scale

        subifu_size = (self.ifu.subifus[0].n_rows * self.ifu.subifus[0].fibre_size) / 1000. * uu.mm

        scale_deg = (self._scale * subifu_size).to('deg').value

        scale_ra = scale_deg / np.cos(np.radians(self.coords.dec.deg))
        scale_dec = scale_deg

        self.ifu.translate(self.coords.ra.deg, self.coords.dec.deg)

        centre_point = shapely.geometry.Point(self.coords.ra.deg, self.coords.dec.deg)
        self.ifu.scale(scale_ra, scale_dec, origin=centre_point)

    def plot(self, ax, show_gap_centres=True, **kwargs):
        """Plots the tile.

        This method is mostly intended to be called by `.Tiling.plot`.

        Parameters:
            ax (`~matplotlib.axes.Axes`):
                The matplotlib axes on which the tile will be plotted.
            show_gap_centres (bool):
                If `True`, the centres of the IFU gaps will be shown
                as dots in the plot.
            kwargs (dict):
                Matplotlib keywords to be passed to the tile
                `~matplotlib.patches.Polygon` to customise its style.

        """

        if show_gap_centres and len(self.ifu.gap_centres) > 0:
            ax.scatter(self.ifu.gap_centres[:, 0], self.ifu.gap_centres[:, 1],
                       color='k', marker='o', facecolor='k', edgecolor='k', zorder=100)

        for subifu in self.ifu.subifus:
            plot_kw = {'facecolor': 'None', 'edgecolor': 'k', 'lw': 1}
            plot_kw.update(kwargs)
            subifu_patch = matplotlib.patches.Polygon(subifu.polygon.exterior.coords, **plot_kw,
                                                      zorder=10)
            ax.add_patch(subifu_patch)

        ax.autoscale_view()

        return ax


class Tiling(object):
    """Performs tiling on a target on the sky.

    Parameters:
        target (`.Target`):
            The `.Target` to be tiled.
        telescope (`~lvmsurveysim.telescope.Telescope`):
            The telescope that will be used to tile the target.

    Attributes:
        tiles (list):
            A list of `.Tile` objects describing the optimal tile for the
            target.

    Example:
        When a `.Tiling` is instatiated with a `.Target` and an `.IFU`, the
        tiling process is run on init ::

            >>> apo1 = Telescope('APO-1m')
            >>> m81 = Target.from_target_list('M81')
            >>> mono = MonolithicIFU()
            >>> m81_tiling = Tiling(m81, apo1, ifu=mono)
            >>> m81_tiling.tiles
            [<Tile RA=169.1, Dec=69.2, angle=3.01>, ...]

        Alternatively, a `.Tiling` can be started with just a `.Target`. In
        that case, the tiling is executed when the object is called with a
        valid `.IFU` ::

            >>> m81_tiling = Tiling(m81, apo1)
            >>> m81_tiling.tiles
            []
            >>> m81_tiling(mono)
            >>> m81_tiling.tiles
            [<Tile RA=169.1, Dec=69.2, angle=3.01>, ...]

    """

    def __init__(self, region, telescope):

        raise NotImplementedError('this class is not yet implemented.')

        assert isinstance(region, Region), 'region is of invalid type.'
        self.region = region

        self.telescope = telescope

        self._ifu = None

        self.tiles = []

    def __repr__(self):

        return f'<Tiling region={self.region.name!r}, tiles={self.tiles!r}>'

    def __call__(self, ifu=None, strict=False):
        """Runs the tiling process using ``ifu`` as the tiling unit.

        Parameters:
            ifu (`.IFU` or None):
                The `.IFU` to be used as tiling unit. If ``None``, the IFU
                can be defined when calling the object. Otherwise, the tiling
                will be run during instantiation.
            strict (bool):
                If `True`, makes sure all the target are gets tiled. Depending
                on the geometry of the IFU this may not be efficient.

        """

        if ifu:
            self._ifu = ifu
        else:
            assert self.ifu

        scale = self.telescope.plate_scale.to('degree/mm')

        tile_centres = self.ifu.get_tile_grid(self.target, scale)

        possible_tiles = []

    def plot(self, **kwargs):
        """Plots the tiles on the target region.

        Parameters:
            kwargs (dict):
                Keyword arguments to be passed to `.Target.plot`. The style
                of each `.Tile` can be configured by calling
                `.Tile.set_plot_params`.

        Returns:
            ax:
                The matplotlib `~matplotlib.axes.Axes` object for this plot.

        """

        __, ax = self.target.plot()

        scale = self.telescope.plate_scale.to('degree/mm')

        tile_centres = self.ifu.get_tile_grid(self.target, scale)

        ax.scatter(tile_centres[:, 0], tile_centres[:, 1], marker='.', color='k', zorder=100)

        return ax

    @property
    def ifu(self):
        """Returns the `.IFU` used for tiling."""

        if self._ifu is None:
            raise ValueError('ifu has not yet been defined. '
                             'Use Tiling(ifu) to set it in runtime.')

        return self._ifu
