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

from astropy import coordinates as coo

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
        angle (`~astropy.coordinates.Angle` or float):
            The rotation angle of the `.IFU` measured from North to East.
        plot_params (dict):
            A dictionary of matplotlib keywords to be used when plotting the
            Tile.

    """

    def __init__(self, ifu, coords, angle=0, plot_params=None):

        assert isinstance(ifu, IFU), 'ifu is not a valid input type.'
        self.ifu = ifu

        if not isinstance(coords, coo.SkyCoord):
            coords = coo.SkyCoord(*coords, unit='deg')

        self.coords = coords
        self.angle = coo.Angle(angle, unit='deg')

        self._plot_params = plot_params

    def __repr__(self):

        return (f'<Tile RA={self.coords.ra.deg:.3f}, '
                f'Dec={self.coords.dec.deg:.3f}, '
                f'angle={self.angle.deg:.1f}>')

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

            >>> m81 = Target.from_target_list('M81')
            >>> mono = MonolithicIFU()
            >>> m81_tiling = Tiling(m81, ifu=mono)
            >>> m81_tiling.tiles
            [<Tile RA=169.1, Dec=69.2, angle=3.01>, ...]

        Alternatively, a `.Tiling` can be started with just a `.Target`. In
        that case, the tiling is executed when the object is called with a
        valid `.IFU` ::

            >>> m81_tiling = Tiling(m81)
            >>> m81_tiling.tiles
            []
            >>> m81_tiling(mono)
            >>> m81_tiling.tiles
            [<Tile RA=169.1, Dec=69.2, angle=3.01>, ...]

    """

    def __init__(self, target, ifu=None):

        assert isinstance(target, Target), 'target is of invalid type.'
        self.target = target

        self.tiles = []

        # If ifu is defined, we call the tiling routine.
        if ifu is not None:
            self.__call__(ifu)

    def __repr__(self):

        return (f'<Tiling target={self.target.name!r}, tiles={self.tiles!r}>')

    def __call__(self, ifu):
        """Runs the tiling process using ``ifu`` as the tiling unit."""

        self._ifu = ifu

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

        pass

    @property
    def ifu(self):
        """Returns the `.IFU` used for tiling."""

        if self._ifu is None:
            raise ValueError('ifu has not yet been defined. '
                             'Use Tiling(ifu) to set it in runtime.')

        return self._ifu
