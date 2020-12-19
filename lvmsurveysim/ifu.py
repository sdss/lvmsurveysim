#!/usr/bin/env python
# encoding: utf-8
#
# ifu.py
#
# Created by José Sánchez-Gallego on 5 Sep 2017.


from __future__ import absolute_import, division, print_function

import matplotlib.collections
import matplotlib.patches
import matplotlib.pyplot
import numpy
import seaborn
import shapely.geometry
import astropy.units

import lvmsurveysim.utils.geodesic_sphere
import lvmsurveysim
from lvmsurveysim import config

seaborn.set()
current_palette = seaborn.color_palette()


__all__ = ['IFU']


def fibres_to_rows(fibres):
    """Calculates the number of rows for a set of fibres.

    Given a total number of ``fibres``, returns the number rows. Given that the
    IFU is an hexagon, the central row has as many fibres as rows in the IFU.
    It returns ``None`` if the number of fibres can not be arranged in a
    full IFU.

    """

    tmp_fibres = 0
    n_central = 1

    while tmp_fibres <= fibres:

        n_central += 2  # Central row needs to be odd

        n_rows = int(1 + 2 * numpy.floor(n_central / 2.))
        tmp_fibres = n_central + 2 * numpy.sum(numpy.arange(n_central - 1, n_central / 2., -1))

        if tmp_fibres == fibres:
            return n_rows


class Fibre(object):

    def __init__(self, xx, yy, radius):

        self.x = xx
        self.y = yy
        self.radius = radius

    @property
    def patch(self):

        return matplotlib.patches.Circle((self.x, self.y), radius=self.radius,
                                         edgecolor='k', facecolor='None', lw=1)


class SubIFU(object):
    """Represents a sub-IFU (group of contiguous fibres) within a larger IFU.

    A sub-IFU is an hexagonal group of contiguous fibres, separated from other
    groups of fibres. LVM IFUs are formed by a series of sub-IFUs (a
    monolithic IFU can be considered as an IFU with a single sub-IFU). It is
    assumed that all sub-IFUs in an IFU have the same size.

    Parameters:
        n_subifu (int):
            An integer used to identify the sub-IFU.
        parent (:class:`IFU` object)
            The parent :class:`IFU` object.
        centre (tuple):
            A 2D tuple describing the centre of the sub-IFUs. It asumes the
            diameter of the sub-IFU hexagon is 1.
        n_fibres (int):
            Number of fibres in the sub-IFUs.
        fibre_size (float):
            The real size, in microns, of each fibre including buffer.

    """

    def __init__(self, id_subifu, parent, centre, n_fibres, fibre_size=None):
        assert isinstance(centre, (list, tuple, numpy.ndarray)), 'centre is not a list'
        #assert isinstance(parent, IFU), 'parent must be an IFU object.'
        assert len(centre) == 2, 'centre must be a 2D tuple.'

        assert n_fibres is not None, 'incorrect inputs.'

        self.id_subifu = id_subifu
        self.parent = parent

        self.n_fibres = n_fibres
        self.n_rows = fibres_to_rows(self.n_fibres)

        self.centre = numpy.array(centre)

        self.fibre_size = fibre_size
        if not isinstance(self.fibre_size, astropy.units.Quantity):
            if not self.fibre_size:
                self.fibre_size = config['fibre_size']
            self.fibre_size *= astropy.units.micron

        self.polygon = self._create_polygon()
        self.fibres = self._create_fibres()

    def _create_polygon(self):
        """Creates a Shapely Polygon collection representing the sub-IFU."""

        RR = 0.5                   # Assumes unitary diameter
        rr = numpy.sqrt(3) / 2. * RR  # Inner radius
        cos60 = 0.5

        xx, yy = self.centre
        vertices = [(xx - RR, yy),
                    (xx - RR * cos60, yy + rr),
                    (xx + RR * cos60, yy + rr),
                    (xx + RR, yy),
                    (xx + RR * cos60, yy - rr),
                    (xx - RR * cos60, yy - rr)]
        polygon = shapely.geometry.Polygon(vertices)

        return polygon

    def _create_fibres(self):
        """Creates Fibre objects for each of the fibres in the sub-IFU."""

        n_centre = self.n_rows
        if self.n_rows is None:
            raise ValueError('incorrect number if fibres. Cannot create a sub-IFU.')

        fibres = []
        fibre_width = 1 / n_centre

        for row in range(int(-self.n_rows / 2), int(self.n_rows / 2) + 1):
            n_fibres_row = n_centre - numpy.abs(row)
            y_row = row * numpy.sqrt(3) / 2 / self.n_rows
            row_length = 1 - 2 * numpy.abs(y_row) * numpy.sqrt(3) / 3.

            fibres_x = (numpy.arange(n_fibres_row) * fibre_width -
                        row_length / 2. + fibre_width / 2.)

            fibres += [Fibre(self.centre[0] + fibre_x,
                             self.centre[1] + y_row,
                             fibre_width / 2.) for fibre_x in fibres_x]

        return fibres

    def scale(self, hscale, vscale, origin='center'):
        """Scales the sub-IFU."""

        self.polygon = shapely.affinity.scale(self.polygon, hscale, vscale, origin=origin)
        self.centre = numpy.array(self.polygon.centroid.coords)[0]

    def translate(self, hor, ver):
        """Translates the IFU."""

        self.polygon = shapely.affinity.translate(self.polygon, hor, ver)
        self.centre = numpy.array(self.polygon.centroid.coords)[0]

    def get_patch(self, scale=None, centre=None, **kwargs):
        """Returns a matplotlib patch for the sub-IFU.

        Parameters
        ----------
        scale : ~astropy.units.Quantity or float
            The plate scale to be used to convert the IFU to on-sky distances.
            Either a `~astropy.units.Quantity` or a value in degrees/mm.
        centre : list
            The coordinates of the centre of the IFU on the sky.
        kwargs : dict
            Parameters to be passed to `~matplotlib.patches.Polygon` when
            creating the patch.

        Returns
        -------
        path : `~matplotlib.patches.Polygon`
            A Matplotlib patch with the sub-ifu. If scale and centre are
            passed, the coordinates of the patch are on-sky.

        """

        if scale is not None and isinstance(scale, astropy.units.Quantity):
            scale = scale.to('degree/mm').value

        vertices = numpy.array(self.polygon.exterior.coords)

        if scale:
            # Calculates the radius in degrees on the sky
            rr_deg = self.n_rows * self.fibre_size.to('mm').value * scale / 2.
            vertices *= rr_deg * 2

        if centre:
            assert scale is not None, 'cannot define a centre without scale.'
            centre = numpy.array(centre)

            # Calculate declinations first so that we can scale RA for
            # all the vertices.
            decs = vertices[:, 1] + centre[1]
            ras = vertices[:, 0] / numpy.cos(numpy.deg2rad(decs)) + centre[0]

            vertices = numpy.array([ras, decs]).T

        return matplotlib.patches.Polygon(vertices, **kwargs)

    def get_patch_collection(self, ax):
        """Returns a collection of fibre patches."""

        n_centre = self.n_rows
        fibre_width = 1 / n_centre * 0.8  # 0.8 to make plotting look nicer.
        fibres = self.fibres
        fibres_patch = matplotlib.collections.EllipseCollection(
            [fibre_width] * len(fibres), [fibre_width] * len(fibres), [0] * len(fibres),
            offsets=[[fibre.x, fibre.y] for fibre in fibres],
            transOffset=ax.transData,
            lw=1, edgecolor='k', facecolor='None',
            units='x')

        return fibres_patch


class IFU(object):
    """A generic class representing a LVM hexagonal IFU.

    This class is intended to be subclassed into real examples of IFU
    designs that LVM will use. An ``IFU`` is defined by a series of sub-IFU
    centres (or a single one for a monolithic IFU) and a fibre size, so
    that the real size of the IFU on the sky can be calculated.

    Parameters:
        n_fibres (int):
            Number of fibres in each of the sub-IFUs.
        centres (list):
            A list of 2D tuples describing the centres of each of the
            sub-IFUs. It assumes the diameter of the sub-IFU hexagon is 1.
        padding (int):
            Number of fibres the IFU should overlap when tiling. This will
            be used to slightly modify the distance between sub-IFUs.
        fibre_size (float):
            The real size, in microns, of each fibre including buffer.
        gaps (list):
            If the IFU is composed of multiple sub-IFUs and gaps exist
            between them, a list with the same format of ``centres`` must
            be provided, with the list of the hexagonal gap centres.
        allow_rotation (bool):
            Whether this IFU can be rotated.

    """

    def __init__(self, n_fibres=None, centres=None, padding=0,
                 fibre_size=None, allow_rotation=False, name=None):

        assert isinstance(centres, (list, tuple, numpy.ndarray)), 'centres is not a list'
        assert len(centres) > 0, 'centres must be a non-zero length list.'
        for ll in centres:
            assert isinstance(ll, (list, tuple, numpy.ndarray)), 'each centre must be a 2D tuple.'
            assert len(ll) == 2, 'each centre must be a 2D tuple.'

        assert n_fibres is not None, 'incorrect n_fibres input.'

        self.name = name

        self.centres = numpy.atleast_2d(centres)

        self.n_fibres = n_fibres
        self.n_subifus = self.centres.shape[0]
        assert self.n_fibres % self.n_subifus == 0, \
            'number of fibres is not a multiple of number of sub-IFUs.'

        self.fibre_size = fibre_size
        if not isinstance(self.fibre_size, astropy.units.Quantity):
            if not self.fibre_size:
                self.fibre_size = config['fibre_size']
            self.fibre_size *= astropy.units.micron

        self.padding = padding

        self.subifus = self._create_subifus()
        self.polygon = shapely.geometry.MultiPolygon([subifu.polygon for subifu in self.subifus])

        self.allow_rotation = allow_rotation

    def __repr__(self):

        return f'<IFU (name={self.name!r}, n_fibres={self.n_fibres}, centres={self.centres!s})>'

    @classmethod
    def from_config(cls):
        """Returns an `.IFU` object from the configuration file."""

        ifu_conf = config['ifu'].copy()

        name = ifu_conf.pop('type', None)

        return cls(name=name, **ifu_conf)

    def scale(self, hscale, vscale, origin='center'):
        """Scales the IFU."""

        for subifu in self.subifus:
            subifu.scale(hscale, vscale, origin=origin)

    def translate(self, hor, ver):
        """Translates the IFU."""

        for subifu in self.subifus:
            subifu.translate(hor, ver)

    def _create_subifus(self):
        """Creates each one of the individual sub-IFUs in this IFU."""

        subifus = []
        n_subifu = 1
        for centre in self.centres:
            subifus.append(SubIFU(n_subifu, self, centre, self.n_fibres / self.n_subifus,
                                  fibre_size=self.fibre_size))
            n_subifu += 1

        return subifus

    def get_patch(self, **kwargs):
        """Returns a matplotlib patch for the IFU.

        Parameters
        ----------
        kwargs : dict
            Parameters to be passed to `.SubIFU.get_patch`.

        """

        return [subifu.get_patch(**kwargs) for subifu in self.subifus]

    def get_tile_grid(self, region, scale, sparse=None, geodesic=None):
        """Returns a grid of positions that tile a region with this IFU.

        Parameters
        ----------
        region : ~shapely.geometry.polygon.Polygon
            The Shapely region to tile. It is assumed that x coordinates are RA
            and y is Declination, both in degrees.
        scale : float
            The scale in degrees per mm.
        sparse : float
            Factor for sparse sampling. Stretches IFU length scale by the number.
        geodesic : use geodesic sphere tiling, sparse gives depth in this case.
        """

        if isinstance(scale, astropy.units.Quantity):
            scale = scale.to('degree/mm').value

        if isinstance(region, lvmsurveysim.target.Region):
            region_shapely = region.shapely
        elif isinstance(region, shapely.geometry.Polygon):
            region_shapely = region
        else:
            raise ValueError(f'invalid region type: {type(region)}.')

        points = []
        # Calculates the radius and apotheme of each subifu in degrees on the sky
        sparse = sparse if sparse!=None else 1.0
        n_rows = self.subifus[0].n_rows
        rr_deg = n_rows * self.fibre_size / 1000 * scale / 2. * sparse
        aa_deg = numpy.sqrt(3) / 2. * rr_deg

        if geodesic == False:
            # Determine the centroid and bounds of the region
            centroid = numpy.array(region_shapely.centroid)
            ra0, dec0, ra1, dec1 = region_shapely.bounds

            # The size of the grid in RA and Dec, in degrees.
            size_ra  = numpy.abs(ra1 - ra0) * numpy.cos(numpy.radians(centroid[1]))
            size_dec = numpy.abs(dec1 - dec0)

            # The separation between grid points in RA and Dec
            delta_ra = 3 * rr_deg
            delta_dec = aa_deg

            # Calculates the initial positions of the grid points in RA and Dec.
            ra_pos = numpy.arange(-size_ra / 2., size_ra / 2. + delta_ra.value, delta_ra.value)
            dec_pos = numpy.arange(-size_dec / 2., size_dec / 2. + delta_dec.value, delta_dec.value)
            points = numpy.zeros((len(dec_pos), len(ra_pos), 2))

            # Offset each other row in RA by 1.5R
            points[:, :, 0] = ra_pos
            points[:, :, 0][1::2] += (1.5 * rr_deg.value)

            # Set declination values
            points[:, :, 1] = dec_pos[numpy.newaxis].T
            points[:, :, 1] += centroid[1]

            # The separations in the RA axis must be converted to RA using the
            # local declination
            points[:, :, 0] /= numpy.cos(numpy.radians(points[:, :, 1]))
            points[:, :, 0] += centroid[0]

            # Reshape into a 2D list of points.
            points = points.reshape((-1, 2))
        else:
            s = lvmsurveysim.utils.geodesic_sphere.initialize_sphere(int(sparse))
            x, y, z = lvmsurveysim.utils.geodesic_sphere.vecs_to_lists(s)
            sk = astropy.coordinates.SkyCoord(x=x,y=y,z=z, representation_type='cartesian')
            sk.representation_type='spherical'
            points = numpy.zeros((len(sk),2))
            points[:,0] = sk.ra.deg
            points[:,1] = sk.dec.deg

        print(geodesic, len(points))

        # For each grid position create a Shapely circle with the radius of the IFU.
        points_shapely = list(
            map(lambda point: shapely.geometry.Point(point[0],
                                                     point[1]).buffer(2. * rr_deg.value),
                points))

        # Check what grid points would overlap with the region if occupied by an IFU.
        inside = list(map(region_shapely.intersects, points_shapely))
        points_inside = points[inside]

        return points_inside

    def plot(self, show_fibres=False, filled=True):
        """Plots the IFU."""

        with seaborn.axes_style('white'):

            fig, ax = matplotlib.pyplot.subplots()

            for subifu in self.subifus:
                ax.add_patch(subifu.get_patch(filled=filled))

                if show_fibres:
                    ax.add_collection(subifu.get_patch_collection(ax))

            ax.autoscale_view()

            # Pads the xy limits
            bounds = shapely.geometry.MultiPolygon([subifu.polygon
                                                    for subifu in self.subifus]).bounds

            xx_pad = 0.1 * (bounds[2] - bounds[0])
            yy_pad = 0.1 * (bounds[3] - bounds[1])

            ax.set_xlim(bounds[0] - xx_pad, bounds[2] + xx_pad)
            ax.set_ylim(bounds[1] - yy_pad, bounds[3] + yy_pad)

        return fig
