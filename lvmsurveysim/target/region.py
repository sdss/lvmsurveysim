#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 17, 2017
# @Filename: region.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import absolute_import, division, print_function

import abc

import astropy.coordinates
import astropy.units
import matplotlib.patches
import matplotlib.path
import matplotlib.transforms
import numpy
import shapely.affinity
import shapely.geometry

from . import plot as lvm_plot
from ..utils import add_doc


__all__ = ['Region', 'EllipticalRegion', 'RectangularRegion',
           'OverlapRegion', 'CircularRegion', 'PolygonalRegion']


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
            region = EllipticalRegion(*args[1:], **kwargs)
        elif args[0] == 'circle':
            region = CircularRegion(*args[1:], **kwargs)
        elif args[0] == 'rectangle':
            region = RectangularRegion(*args[1:], **kwargs)
        elif args[0] == 'polygon':
            region = PolygonalRegion(*args[1:], **kwargs)
        else:
            raise ValueError('invalid region type.')

        region.region_type = args[0]

        return region

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

        self._shapely = None
        self.frame = self.frame or 'icrs'

    def __repr__(self):

        return f'<{self.__class__.__name__}>'

    @abc.abstractmethod
    def _create_shapely(self):
        """Creates the `Shapely`_ object representing this region.

        To be overridden by subclasses.

        """

        pass

    @property
    def shapely(self):
        """Returns the `Shapely`_ representation of the ellipse."""

        if self._shapely is None:
            self._shapely = self._create_shapely()

        return self._shapely

    @abc.abstractmethod
    def plot(self, projection='rectangular', **kwargs):
        """Plots the region.

        Parameters:
            projection (str):
                The projection to use. At this time, only ``rectangular`` and
                ``mollweide`` are accepted.
            return_patch (bool):
                If True, returns the
                `matplotlib patch <https://matplotlib.org/api/patches_api.html>`_
                for the region.
            kwargs (dict):
                Options to be passed to matplotlib when creating the patch.

        Returns:
            Returns the matplotlib `~matplotlib.figure.Figure` and
            `~matplotlib.axes.Axes` objects for this plot. If not specified,
            the default plotting styles will be used. If ``return_patch=True``,
            returns the patch as well.

        """

        pass

    def contains(self, coords):
        """Returns ``True`` if the ``coords`` are inside the region.

        Parameters:
        coords (tuple or `~astropy.coordinates.SkyCoord`):
            A tuple of ``(coord1, coord2)`` in degrees or the
            `~astropy.coordinates.SkyCoord` the point to test.

        """

        if isinstance(coords, astropy.coordinates.SkyCoord):
            point = shapely.geometry.Point((coords.data.lon.deg, coords.data.lat.deg))
        else:
            point = shapely.geometry.Point((coords[0], coords[1]))

        return self.shapely.contains(point)

    def overlap(self, other):
        """Returns the `.OverlapRegion` between this and other sky region.

        Uses shapely to determine the intersection between two regions and
        creates an overlap region. Returns ``False`` if there is no
        intersection.

        """

        assert isinstance(other, Region), 'other must be a subclass of Region.'

        if self.shapely.intersects(other.shapely) is False:
            return False

        return OverlapRegion(self.shapely.intersection(other.shapely))

    def to_cartesian(self):
        """Converts from polar to cartesian coordinates."""

        exterior = list(self.shapely.exterior.coords)

        exterior_coords = astropy.coordinates.SkyCoord(exterior, frame=self.frame, unit='deg')
        exterior_icrs = exterior_coords.icrs

        return astropy.coordinates.spherical_to_cartesian(
            1,
            numpy.deg2rad(exterior_icrs.data.lat.value),
            numpy.deg2rad(exterior_icrs.data.lon.value))

    def to_healpix(self, pixarea=None, ifu=None, telescope=None, return_coords=False):
        """Tessellates the target region and returns a list of HealPix cells.

        Parameters
        ----------
        pixarea : float
            Desired area of the HealPix cell, in square degrees. The HealPix
            order that produces a cell of size equal or smaller than
            ``pixarea`` will be used.
        ifu : `~lvmsurveysim.tiling.IFU`
            The IFU used for tiling the region.
        telescope : `~lvmsurveysim.telescope.Telescope`
            The telescope on which the IFU is mounted.

        """

        import healpy

        assert pixarea is not None or ifu is not None or telescope is not None, \
            'either pixarea or ifu and telescope need to be defined.'

        if pixarea is None:
            assert ifu and telescope, 'ifu and telescope need to be defined.'
            raise NotImplementedError

        order = 0
        while order <= 30:
            if healpy.pixelfunc.nside2pixarea(2**order, degrees=True) <= pixarea:
                break
            order += 1

        if order == 30:
            raise ValueError('pixarea is too small.')

        pixels = healpy.query_polygon(
            2**order, numpy.array(self.to_cartesian()).T[:-2], inclusive=False)

        if return_coords:
            return numpy.array(healpy.pixelfunc.pix2ang(2**order, pixels, lonlat=True)).T
        return pixels


class EllipticalRegion(Region):
    """A class that represents an elliptical region on the sky.

    Represents an elliptical region. Internally it is powered by a shapely
    `Point <http://toblerity.org/shapely/manual.html#Point>`_ object.

    Parameters:
        coords (tuple or `~astropy.coordinates.SkyCoord`):
            A tuple of ``(coord1, coord2)`` in degrees or a
            `~astropy.coordinates.SkyCoord` describing the centre of the
            ellipse.
        a (float or `~astropy.coordinates.Angle`):
            The length of the semi-major axis of the ellipse, in degrees.
            Gets converted to an `~astropy.coordinates.Angle`.
        b (float or `~astropy.coordinates.Angle`):
            The length of the semi-minor axis of the ellipse, in degrees. Gets
            converted to an `~astropy.coordinates.Angle`.
        ba (float):
            The ratio of the semi-minor to semi-major axis. Either ``b`` or
            ``ba`` must be defined.
        pa (float or `~astropy.coordinates.Angle`):
            The position angle (from North to East) of the major axis of the
            ellipse, in degrees. Gets converted to an
            `~astropy.coordinates.Angle`.
        frame : str
            The coordinate frame in which the coordinates are. E.g.,
            ``'icrs'`` or ``'galactic'``. Must be one of the frames understood
            by `~astropy.coordinates.SkyCoord`.

    """

    def __init__(self, coords, a, b=None, pa=None, ba=None, frame=None):

        self.a = astropy.coordinates.Angle(a, 'deg')

        assert pa is not None, 'position angle is missing.'

        self.pa = astropy.coordinates.Angle(pa, 'deg')

        if b is None:
            assert ba is not None, 'either b or ba need to be defined.'
            self.b = astropy.coordinates.Angle(ba * a, 'deg')
        else:
            self.b = astropy.coordinates.Angle(b, 'deg')

        assert self.a > self.b, 'a must be greater than b.'

        self.frame = frame or 'icrs'

        if not isinstance(coords, astropy.coordinates.SkyCoord):
            assert len(coords) == 2, 'invalid number of coordinates.'
            self.coords = astropy.coordinates.SkyCoord(coords[0],
                                                       coords[1],
                                                       frame=frame,
                                                       unit='deg')

        super(EllipticalRegion, self).__init__()

        if self.frame != 'icrs':
            raise NotImplementedError('EllipticalRegion only accepts frame="icrs".')

    def __repr__(self):

        return (f'<{self.__class__.__name__} (coords={self.coords!r}, '
                f'a={self.a:.3f}, b={self.b:.3f}, pa={self.pa:.1f})>')

    def _create_shapely(self, a=None, b=None, pa=None):
        """Creates a `Shapely`_ object representing the ellipse."""

        a = a if a is not None else self.a
        b = b if b is not None else self.b
        pa = pa if pa is not None else self.pa

        a = astropy.coordinates.Angle(a, 'deg')
        b = astropy.coordinates.Angle(b, 'deg')
        pa = astropy.coordinates.Angle(pa, 'deg')

        # See https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely/243462

        circ = shapely.geometry.Point((self.coords.data.lon.deg,
                                       self.coords.data.lat.deg)).buffer(1)

        ell = shapely.affinity.scale(circ, b.deg, a.deg)

        # Rotate the ellipse.
        ellr = shapely.affinity.rotate(ell, -pa.deg)

        # Applies the RA axis scaling

        ell_lon = shapely.affinity.scale(ellr,
                                        1 / numpy.cos(numpy.radians(self.coords.data.lat.deg)), 1)

        return ell_lon

    @add_doc(Region.plot)
    def plot(self, projection='rectangular', return_patch=False,
             a=None, b=None, pa=None, **kwargs):

        a = a if a is not None else self.a
        b = b if b is not None else self.b
        pa = pa if pa is not None else self.pa

        a = astropy.coordinates.Angle(a, 'deg')
        b = astropy.coordinates.Angle(b, 'deg')
        pa = astropy.coordinates.Angle(pa, 'deg')

        fig, ax = lvm_plot.get_axes(projection=projection)

        # Creates the ellipse in (0, 0) so that the scaling doesn't move the
        # centre. We will translate it after scaling in the RA direction.
        ell = matplotlib.patches.Ellipse((0, 0),
                                         width=b.deg * 2,
                                         height=a.deg * 2,
                                         angle=-pa.deg, **kwargs)

        ell = ax.add_patch(ell)

        # Scales the RA direction by the cos(dec) factor. We want to do this
        # AFTER the rotation has happened, so that all the elements in the RA
        # direction are scaled properly.
        lon_transform = matplotlib.transforms.Affine2D().scale(
            1. / numpy.cos(numpy.radians(self.coords.data.lat.deg)), 1)

        # Moves the ellipse to the correct position.
        coords_transform = matplotlib.transforms.Affine2D().translate(self.coords.data.lon.deg,
                                                                      self.coords.data.lat.deg)

        # This way of applying the transformation makes sure ra_transform
        # is applied in data units before ax.transData converts to pixels.
        ell.set_transform(lon_transform + coords_transform + ax.transData)

        if projection == 'rectangular':

            padding_x = 0.1 * (self.shapely.bounds[2] - self.shapely.bounds[0])
            padding_y = 0.1 * (self.shapely.bounds[3] - self.shapely.bounds[1])

            ax.set_xlim(self.shapely.bounds[2] + padding_x, self.shapely.bounds[0] - padding_x)
            ax.set_ylim(self.shapely.bounds[1] - padding_y, self.shapely.bounds[3] + padding_y)

        elif projection == 'mollweide':

            centre = ell.center[0]
            ell = lvm_plot.transform_patch_mollweide(ax, ell, patch_centre=centre)

        if return_patch:
            return fig, ax, ell
        else:
            return fig, ax


class CircularRegion(EllipticalRegion):
    """A class that represents a circular region on the sky.

    Represents a circular region. Internally it is powered by a shapely
    `Point <http://toblerity.org/shapely/manual.html#Point>`_ object.

    Parameters:
        coords (tuple or `~astropy.coordinates.SkyCoord`):
            A tuple of ``(coord1, coord2)`` in degrees or a
            `~astropy.coordinates.SkyCoord` describing the centre of the
            circle.
        r (float):
            The radius of the circle, in degrees.
        frame : str
            The coordinate frame in which the coordinates are. E.g.,
            ``'icrs'`` or ``'galactic'``. Must be one of the frames understood
            by `~astropy.coordinates.SkyCoord`.

    """

    def __init__(self, coords, r, frame=None):

        self.r = astropy.coordinates.Angle(r, 'deg')
        self.frame = frame or 'icrs'

        if not isinstance(coords, astropy.coordinates.SkyCoord):
            assert len(coords) == 2, 'invalid number of coordinates.'
            self.coords = astropy.coordinates.SkyCoord(coords[0],
                                                       coords[1],
                                                       frame=frame,
                                                       unit='deg')

        Region.__init__(self)

    def __repr__(self):

        return f'<{self.__class__.__name__} (coords={self.coords!r}, r={self.r:.3f})>'

    def _create_shapely(self):
        """Creates a `Shapely`_ object representing the ellipse."""

        circ = super(CircularRegion, self)._create_shapely(a=self.r.deg,
                                                           b=self.r.deg,
                                                           pa=0)

        return circ

    @add_doc(Region.plot)
    def plot(self, projection='rectangular', return_patch=False, **kwargs):

        return super(CircularRegion, self).plot(projection=projection,
                                                return_patch=return_patch,
                                                a=self.r, b=self.r, pa=0, **kwargs)


class PolygonalRegion(Region):
    """Defines a polygonal (multipoint) region on the sky.

    Represents a polygon on the sky Internally it is powered by a shapely
    `Polygon <https://shapely.readthedocs.io/en/latest/manual.html#polygons>`_
    object.

    Parameters
    ----------
    vertices : list
            A list or `~numpy.ndarray` of the vertices of the polygon, each
            one of them a tuple ``(coord1, coord2)``. If the last element is
            not identical to the first one, the polygon is closed using the
            first vertex.
    pa : float
        A rotation (North to East) to be applied to the vertices.
    frame : str
        The coordinate frame in which the coordinates are. E.g.,
        ``'icrs'`` or ``'galactic'``. Must be one of the frames understood
        by `~astropy.coordinates.SkyCoord`.

    Example
    -------

        >>> poly = PolygonalRegion([(169, 65), (180, 65), (170, 70), (169, 65)])

    """

    def __init__(self, vertices, pa=0.0, frame=None):

        self._orig_vertices = numpy.atleast_2d(vertices).astype(numpy.float)
        self.frame = frame

        assert self._orig_vertices.ndim == 2, 'invalid number of dimensions.'
        assert self._orig_vertices.shape[0] > 2, 'need at least three points for a polygon.'

        Region.__init__(self)

    def __repr__(self):

        return f'<{self.__class__.__name__} (vertices={self.vertices!r})>'

    def _create_shapely(self):
        """Creates a `Shapely`_ object representing the ellipse."""

        poly = shapely.geometry.Polygon(self._orig_vertices.tolist())

        return poly

    @add_doc(Region.plot)
    def plot(self, projection='rectangular', return_patch=False, **kwargs):

        fig, ax = lvm_plot.get_axes(projection=projection)

        poly = matplotlib.path.Path(self.vertices, closed=True)
        poly_patch = matplotlib.patches.PathPatch(poly, **kwargs)

        poly_patch = ax.add_patch(poly_patch)

        if projection == 'rectangular':

            padding_x = 0.1 * (self.shapely.bounds[2] - self.shapely.bounds[0])
            padding_y = 0.1 * (self.shapely.bounds[3] - self.shapely.bounds[1])

            ax.set_xlim(self.shapely.bounds[2] + padding_x, self.shapely.bounds[0] - padding_x)
            ax.set_ylim(self.shapely.bounds[1] - padding_y, self.shapely.bounds[3] + padding_y)

        elif projection == 'mollweide':

            centre = self.shapely.centroid.x
            poly_patch = lvm_plot.transform_patch_mollweide(ax, poly_patch, patch_centre=centre)

        if return_patch:
            return fig, ax, poly_patch
        else:
            return fig, ax


class RectangularRegion(PolygonalRegion):
    """A class that represents a rectangular region on the sky.

    Creates a rectangular region for a rectangle on the sky.

    Parameters
    ----------
    coords : `tuple` or `~astropy.coordinates.SkyCoord`
        A tuple of ``(coord1, coord2)`` in degrees or a
        `~astropy.coordinates.SkyCoord` describing the centre of the rectangle.
    width : float
        The length along the first axis of the rectangle, in degrees.
    height : float
        The length along the second axis of the rectangle, in degrees.
    pa : float
        The position angle (North to East) of the rectangle.
    frame : str
        The coordinate frame in which the coordinates are. E.g., ``'icrs'`` or
        ``'galactic'``. Must be one of the frames understood by
        `~astropy.coordinates.SkyCoord`.

    """

    def __init__(self, coords, width, height, pa=0.0, frame=None):

        if not isinstance(coords, astropy.coordinates.SkyCoord):
            assert len(coords) == 2, 'invalid number of coordinates.'
            self.coords = astropy.coordinates.SkyCoord(coords[0],
                                                       coords[1],
                                                       frame=frame or 'icrs',
                                                       unit='deg')
        self.width = width * astropy.units.degree
        self.height = height * astropy.units.degree
        self.pa = pa * astropy.units.degree

        self.frame = frame or 'icrs'

        width_deg = self.width.value / numpy.cos(numpy.deg2rad(self.coords.data.lat.value))
        height_deg = self.height.value

        x0 = self.coords.data.lon.value - width_deg / 2.
        x1 = self.coords.data.lon.value + width_deg / 2.
        y0 = self.coords.data.lat.value - height_deg / 2.
        y1 = self.coords.data.lat.value + height_deg / 2.

        PolygonalRegion.__init__(self,
                                 vertices=[(x0, y0), (x0, y1), (x1, y1), (x1, y0)],
                                 pa=self.pa.value,
                                 frame=self.frame)

    def __repr__(self):

        return (f'<{self.__class__.__name__} '
                f'(coords={self.coords!r}, width={self.width!s}, '
                f'height={self.height!s}, pa={self.pa!s})>')

    @property
    def vertices(self):
        """Returns the vertices of the rectangle."""

        return numpy.array(self.shapely.exterior.coords.xy).T

    def _create_shapely(self):
        """Creates a `Shapely`_ object representing the rectangle."""

        width_deg = self.width.value / numpy.cos(numpy.deg2rad(self.coords.data.lat.value))
        height_deg = self.height.value

        x0 = self.coords.data.lon.value - width_deg / 2.
        x1 = self.coords.data.lon.value + width_deg / 2.
        y0 = self.coords.data.lat.value - height_deg / 2.
        y1 = self.coords.data.lat.value + height_deg / 2.

        box = shapely.geometry.box(x0, y0, x1, y1)
        box_rot = shapely.affinity.rotate(box, -self.pa.value)

        return box_rot


class OverlapRegion(Region):
    """Defines the region resulting from the overlap of two regions.

    Not intended for direct instantiation.

    Parameters:
        shapely_region (`shapely.geometry.base.BaseGeometry`):
            The shapely object describing the overlap region.

    """

    def __init__(self, shapely_region):

        super(OverlapRegion, self).__init__()

        self._shapely = shapely_region

    def _create_shapely(self):
        pass

    def _create_patch(self, **kwargs):
        """Returns an `~matplotlib.patches.Ellipse` for this region."""

        raise NotImplementedError

    @add_doc(Region.plot)
    def plot(self, projection='rectangular', **kwargs):

        raise NotImplementedError
