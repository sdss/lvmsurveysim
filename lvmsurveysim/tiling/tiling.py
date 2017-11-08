#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 10, 2017
# @Filename: tiling.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy

import numpy as np

from astropy import coordinates as coo
from astropy import units as uu

import shapely.affinity
import shapely.geometry

from .ifu import IFU
from ..target.target import Target


__all__ = ['Tiling', 'Tile']


class Tile(object):
    """A tiling element.

    A `.Tile` is basically an `.IFU` with a position and a rotation. A
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
        plot_params (dict):
            A dictionary of matplotlib keywords to be used when plotting the
            Tile.

    """

    def __init__(self, ifu, coords, scale, angle=0, plot_params=None):

        assert isinstance(ifu, IFU), 'ifu is not a valid input type.'
        self.ifu = ifu

        if not isinstance(coords, coo.SkyCoord):
            coords = coo.SkyCoord(*coords, unit='deg')

        self.coords = coords
        self.angle = coo.Angle(angle, unit='deg')

        self._plot_params = plot_params

        self.shapely = self._create_shapely(scale)

    def __repr__(self):

        return (f'<Tile RA={self.coords.ra.deg:.3f}, '
                f'Dec={self.coords.dec.deg:.3f}, '
                f'angle={self.angle.deg:.1f}>')

    def _create_shapely(self, scale):
        """Creates a shapely region representing the IFU on the sky."""

        if not isinstance(scale, uu.Quantity):
            scale = scale * uu.arcsec / uu.mm

        subifus_polygons = []

        for subifu in self.ifu.subifus:

            # The radius of each subifu (the central row) is 1 by definition.
            subifu_size = (subifu.n_rows * subifu.n_fibres) / 1000. * uu.mm

            # Scales shapely geometry to mm.
            subifu_mm = shapely.affinity.scale(subifu.geometry,
                                               subifu_size.to('mm').value,
                                               subifu_size.to('mm').value, center=(0, 0))

            # Applies the plate scale and RA correction
            subifu_arcsec = shapely.affinity.scale(
                subifu_mm, scale, scale / np.cos(np.radians(self.coords.dec.deg)),
                center=(0, 0))

            # Translates the IFU to its location on the region.
            subifu_translated = shapely.affinity.translate(subifu_arcsec,
                                                           self.coords.ra.deg,
                                                           self.coords.dec.deg)

            # Rotates the IFU
            sub_ifu_rotated = shapely.affinity.rotate(subifu_translated, -self.angle.deg)

            subifus_polygons.append(sub_ifu_rotated)

        return shapely.geometry.MultiPolygon(subifus_polygons)

    def set_plot_params(self, **kwargs):
        """Sets the default plotting parameters for this tile."""

        self._plot_params = kwargs

    def plot(self, ax, **kwargs):
        """Plots the tile.

        Parameters:
            ax (`~matplotlib.axes.Axes`):
                The matplotlib axes on which the tile will be plotted.
            kwargs (dict):
                Matplotlib keywords to be passed to the tile
                `~matplotlib.patches.Patch` to customise its style. If no
                keywords are passed and ``plot_params`` were defined when
                initialising the `.Tile`, those parameters will be used.
                Otherwise, the default style will be used.

        """

        pass


class Tiling(object):
    """Performs tiling on a target on the sky.

    Parameters:
        target (`.Target`):
            The `.Target` to be tiled.
        telescope (`~lvmsurveysim.telescope.Telescope`):
            The telescope that will be used to tile the target.
        ifu (`.IFU` or None):
            The `.IFU` to be used as tiling unit. If ``None``, the IFU
            can be defined when calling the object. Otherwise, the tiling will
            be run during instantiation.

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

    def __init__(self, target, telescope, ifu=None):

        assert isinstance(target, Target), 'target is of invalid type.'
        self.target = target

        self.telescope = telescope

        self.tiles = []

        # If ifu is defined, we call the tiling routine.
        if ifu is not None:
            self.__call__(ifu)

    def __repr__(self):

        return f'<Tiling target={self.target.name!r}, tiles={self.tiles!r}>'

    def __call__(self, ifu):
        """Runs the tiling process using ``ifu`` as the tiling unit."""

        self._ifu = ifu

        # First pass. We overtile target.
        untiled_shapely = copy.deepcopy(self.target.region.shapely)

        while not untiled_shapely.is_empty:
            repr_point = untiled_shapely.representative_point()
            new_tile = Tile(self.ifu, (repr_point.x, repr_point.y),
                            self.telescope.plate_scale.to('arcsec/mm'))
            self.tiles.append(new_tile)
            untiled_shapely = untiled_shapely.difference(new_tile.ifu.polygon)

    def plot(self, **kwargs):
        """Plots the tiles on the target region.

        Parameters:
            kwargs (dict):
                Keyword arguments to be passed to `.Target.plot`. The style
                of each `.Tile` can be configured by calling
                `.Tile.set_plot_params`.

        Returns:
            fig, ax:
                The matplotlib `~matplotlib.figure.Figure` and
                `~matplotlib.axes.Axes` objects for this plot.

        """

        fig, ax = self.target.plot(**kwargs)

        for tile in self.tiles:
            tile.plot(ax)

        return fig, ax

    @property
    def ifu(self):
        """Returns the `.IFU` used for tiling."""

        if self._ifu is None:
            raise ValueError('ifu has not yet been defined. '
                             'Use Tiling(ifu) to set it in runtime.')

        return self._ifu
