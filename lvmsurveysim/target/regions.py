#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 17, 2017
# @Filename: regions.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import abc

import astropy.coordinates

import shapely.geometry
import shapely.affinity

import numpy as np

import matplotlib.patches

from . import plot as lvm_plot
from ..utils import add_doc


__all__ = ['Region', 'EllipticalRegion']


def region_factory(cls, *args, **kwargs):
    """A factory that returns the right type of region depending on input.

    Based on first argument, determines the type of region to return and
    passes it the ``args`` and ``kwargs``. This function is intended for
    overrding the ``__call__`` method in the `abc.ABCMeta` metacalass. The
    reason is that we want `.Region` to have
    `abstract methods <abc.abstractmethod>` while also being a factory.
    See `this stack overflow <https://stackoverflow.com/a/5961102>`_ for
    details in the implementation of the ``__call__`` factory pattern.

    It can be used as::

        RegionABC = abc.ABCMeta
        RegionABC.__call__ = region_factory


        class Region(object, metaclass=RegionABC):
            ...

    Note that this will override ``__call__`` everywhere else where
    `abc.ABCMeta` is used, but since it only overrides the default behaviour
    when the input class is `.Region`, that should not be a problem.

    """

    if cls is Region:
        if args[0] == 'ellipse':
            return EllipticalRegion(*args[1:], **kwargs)
        elif args[0] == 'circle':
            return CircularRegion(*args[1:], **kwargs)
        else:
            raise ValueError('invalid region type.')

    return type.__call__(cls, *args, **kwargs)


# Overrides the __call__ method in abc.ABC.
RegionABC = abc.ABCMeta
RegionABC.__call__ = region_factory


class Region(object, metaclass=RegionABC):
    """A class describing a region of generic shape on the sky.

    This class acts as both the superclass of all the sky regions, and as
    a factory for them. The first argument `.Region` receives is used to
    determine the type of region to returns. All other arguments and keywords
    are passed to the class for instantiation. See `.region_factory` for
    details on how this behaviour is implemented.

    Parameters:
        region_type (str):
            The type of region to return. Must be one of ``ellipse``,
            ``circle``, or ``polygon``.
        args, kwargs:
            Arguments and keywords arguments that will be passed to the class.

    Example:
        >>> new_region = Region('ellipse', (180, 20), a=0.1, b=0.05, pa=0)
        >>> type(new_region)
        lvmsurveysim.target.regions.EllipticalRegion

    """

    def __init__(self, *args, **kwargs):

        self._shapely = self._create_shapely()

    @abc.abstractmethod
    def _create_shapely(self):
        """Creates the `Shapely`_ object representing this region.

        To be overridden by subclasses.

        """

        pass

    @property
    def shapely(self):
        """Returns the `Shapely`_ representation of the ellipse."""

        return self._shapely

    @abc.abstractmethod
    def plot(self, projection='rectangular', **kwargs):
        """Plots the region.

        Parameters:
            projection (str):
                The projection to use. At this time, only ``rectangular`` and
                ``mollweide`` are accepted.
            kwargs (dict):
                Options to be passed to matplotlib when creating the patch.

        Returns:
            Returns the matplotlib `~matplotlib.figure.Figure` and
            `~matplotlib.axes.Axes` objects for this plot. If not specified,
            the default plotting styles will be used.

        """

        pass


class EllipticalRegion(Region):
    """A class that represents an elliptical region on the sky.

    Represents an elliptical region. Internally it is powered by a shapely
    `Point <http://toblerity.org/shapely/manual.html#Point>`_ object.

    Parameters:
        coords (tuple or `~astropy.coordinates.SkyCoord`):
            A tuple of ``(ra, dec)`` in degrees or a
            `~astropy.coordinates.SkyCoord` describing the centre of the
            ellipse.
        a (float):
            The length of the semi-major axis of the ellipse, in degrees.
        b (float):
            The length of the semi-minor axis of the ellipse, in degrees.
        ba (float):
            The ratio of the semi-minor to semi-major axis. Either ``b`` or
            ``ba`` must be defined.
        pa (float):
            The position angle (from North to East) of the ellipse, in degrees.

    """

    def __init__(self, coords, a, b=None, pa=None, ba=None):

        self.a = a

        assert pa is not None, 'position angle is missing.'

        self.pa = pa

        if b is None:
            assert ba is not None, 'either b or ba need to be defined.'
            self.b = ba * a
        else:
            self.b = b

        if not isinstance(coords, astropy.coordinates.SkyCoord):
            assert len(coords) == 2, 'invalid number of coordinates.'
            self.coords = astropy.coordinates.SkyCoord(ra=coords[0],
                                                       dec=coords[1],
                                                       unit='deg')

        super(EllipticalRegion, self).__init__()

    def _create_shapely(self):
        """Creates a `Shapely`_ object representing the ellipse."""

        # See https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely/243462

        circ = shapely.geometry.Point((self.coords.ra.deg, self.coords.dec.deg)).buffer(1)

        # Create the ellipse along x and y.
        ell = shapely.affinity.scale(circ,
                                     self.a / np.cos(np.deg2rad(self.coords.dec.deg)),
                                     self.b)

        # Rotate the ellipse (positive values mean anticlockwise)
        ellr = shapely.affinity.rotate(ell, self.pa)

        return ellr

    def _create_patch(self, **kwargs):
        """Returns an `~matplotlib.patches.Ellipse` for this region."""

        # Note that Ellipse uses the full length (diameter) of the axes.

        ra = self.coords.ra.deg
        dec = self.coords.dec.deg
        ell = matplotlib.patches.Ellipse((ra, dec),
                                         width=self.a / np.cos(np.deg2rad(dec)) * 2.,
                                         height=self.b * 2.,
                                         angle=self.pa, **kwargs)

        return ell

    @add_doc(Region.plot)
    def plot(self, projection='rectangular', **kwargs):

        fig, ax = lvm_plot.get_axes(projection=projection)
        patch = self._create_patch()
        ax.add_patch(patch)
        ax.autoscale()

        return fig, ax


class CircularRegion(Region):
    """A class that represents a circular region on the sky.

    Represents a circular region. Internally it is powered by a shapely
    `Point <http://toblerity.org/shapely/manual.html#Point>`_ object.

    Parameters:
        coords (tuple or `~astropy.coordinates.SkyCoord`):
            A tuple of ``(ra, dec)`` in degrees or a
            `~astropy.coordinates.SkyCoord` describing the centre of the
            circle.
        r (float):
            The radius of the circle, in degrees.

    """

    def __init__(self, coords, r):

        self.r = r

        if not isinstance(coords, astropy.coordinates.SkyCoord):
            assert len(coords) == 2, 'invalid number of coordinates.'
            self.coords = astropy.coordinates.SkyCoord(ra=coords[0], dec=coords[1], unit='deg')

        super(CircularRegion, self).__init__()

    def _create_shapely(self):
        """Creates a `Shapely`_ object representing the ellipse."""

        # See https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely/243462

        circ = shapely.geometry.Point((self.coords.ra.deg, self.coords.dec.deg)).buffer(self.r)

        return circ

    def _create_patch(self, **kwargs):
        """Returns an `~matplotlib.patches.Ellipse` for this region."""

        # Note that Ellipse uses the full length (diameter) of the axes.

        ra = self.coords.ra.deg
        dec = self.coords.dec.deg
        ell = matplotlib.patches.Ellipse((ra, dec),
                                         width=self.r / np.cos(np.deg2rad(dec)) * 2.,
                                         height=self.r * 2., **kwargs)

        return ell

    @add_doc(Region.plot)
    def plot(self, projection='rectangular', **kwargs):

        fig, ax = lvm_plot.get_axes(projection=projection)
        patch = self._create_patch()
        ax.add_patch(patch)
        ax.autoscale()

        return fig, ax
