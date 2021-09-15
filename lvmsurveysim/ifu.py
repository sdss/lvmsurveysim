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
import astropy.units
from astropy.coordinates.angle_utilities import position_angle

import lvmsurveysim
from lvmsurveysim import config
import lvmsurveysim.utils.geodesic_sphere


# TODO: move get_tile_grid() somewhere else, probably tiledb?

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


class EqTransform(object):
    '''
    The transformation between the two equatorial coordinate systems (for example to galactic):
        1. a rotation around the celestial polar axis by, so that the reference zero longitude matches the node
        2. a rotation around the node by the inclination of the new equator
        3. a rotation around the new polar axis so that the zero new longitude meridian matches the input.

        Parameters:
        RA_NGP : float
            RA coordinate of new north pole in old system
        DEC_NGP : float
            DEC coordinate of new north pole in old system
        L_CP : float
            longitude of the old pole in the new system

        For example: 
            EqTransform(192.8594, 27.1282, 122.9319) 
            transforms from icrs to Galactic coordinates and back.
    '''
    def __init__(self, RA_NGP, DEC_NGP, L_CP):
        self.RA_NGP = numpy.deg2rad(RA_NGP)   # Galactic: numpy.deg2rad(192.8594812065348)
        self.DEC_NGP = numpy.deg2rad(DEC_NGP) # Galactic: numpy.deg2rad(27.12825118085622)
        self.L_CP = numpy.deg2rad(L_CP)       # Galactic: numpy.deg2rad(122.9319185680026)

        self.L_0 = self.L_CP - numpy.pi / 2.
        self.RA_0 = self.RA_NGP + numpy.pi / 2.
        self.DEC_0 = numpy.pi / 2. - self.DEC_NGP

    def eq2gal(self, ra, dec):
        '''
        Forward transform from old to new system
        '''
        ra, dec = numpy.deg2rad(numpy.array(ra, ndmin=1)), numpy.deg2rad(numpy.array(dec, ndmin=1))
        numpy.sinb = numpy.sin(dec) * numpy.cos(self.DEC_0) - numpy.cos(dec) * numpy.sin(ra - self.RA_0) * numpy.sin(self.DEC_0)
        b = numpy.arcsin(numpy.sinb)
        cosl = numpy.cos(dec) * numpy.cos(ra - self.RA_0) / numpy.cos(b)
        sinl = (numpy.sin(dec) * numpy.sin(self.DEC_0) + numpy.cos(dec) * numpy.sin(ra - self.RA_0) * numpy.cos(self.DEC_0)) / numpy.cos(b)
        return self._normalize_angles(cosl, sinl, self.L_0, b)

    def gal2eq(self, l, b):
        '''
        Backwards transform from new to old system
        '''
        l, b = numpy.deg2rad(numpy.array(l, ndmin=1)), numpy.deg2rad(numpy.array(b, ndmin=1))
        sind = numpy.sin(b) * numpy.sin(self.DEC_NGP) + numpy.cos(b) * numpy.cos(self.DEC_NGP) * numpy.sin(l - self.L_0)
        dec = numpy.arcsin(sind)
        cosa = numpy.cos(l - self.L_0) * numpy.cos(b) / numpy.cos(dec)
        sina = (numpy.cos(b) * numpy.sin(self.DEC_NGP) * numpy.sin(l - self.L_0) - numpy.sin(b) * numpy.cos(self.DEC_NGP)) / numpy.cos(dec)
        return self._normalize_angles(cosa, sina, self.RA_0, dec)

    def _normalize_angles(self, cosa, sina, ra0, dec):
        '''
        Make sure everything ends up in the right quadrant
        '''
        dec = numpy.rad2deg(dec)
        cosa[cosa < -1.0] = -1.0
        cosa[cosa > 1.0] = 1.0
        ra = numpy.arccos(cosa)
        ra[numpy.where(sina < 0.)] *= -1.0
        ra = numpy.rad2deg(ra + ra0)
        ra = numpy.mod(ra, 360.)
        dec = numpy.mod(dec + 90., 180.) - 90.
        return ra, dec


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
                self.fibre_size = config['ifu']['fibre_size']
            self.fibre_size *= astropy.units.micron

        self.polygon = self._create_polygon()
        self.fibres = self._create_fibres()

    def _create_polygon(self):
        """Creates a list of polygon vertices representing the shape of this sub-IFU."""

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
        
        return vertices

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


    def get_patch(self, scale=None, centre=None, pa=None, **kwargs):
        """Returns a matplotlib patch for the sub-IFU.

        Parameters
        ----------
        scale : ~astropy.units.Quantity or float
            The plate scale to be used to convert the IFU to on-sky distances.
            Either a `~astropy.units.Quantity` or a value in degrees/mm.
        centre : list
            The coordinates of the centre of the IFU on the sky.
        pa : float
            The position angle of the IFU on the sky in degrees. Ignored if None.
        kwargs : dict
            Parameters to be passed to `~matplotlib.patches.Polygon` when
            creating the patch.

        Returns
        -------
        patch : `~matplotlib.patches.Polygon`
            A Matplotlib patch with the sub-ifu. If scale and centre are
            passed, the coordinates of the patch are on-sky.

        """

        if scale is not None and isinstance(scale, astropy.units.Quantity):
            scale = scale.to('degree/mm').value

        vertices = numpy.array(self.polygon)

        # rotate to posotion angle!
        if pa != None:
            c, s = numpy.cos(numpy.deg2rad(pa)), numpy.sin(numpy.deg2rad(pa))
            x = c*vertices[:, 0] + s*vertices[:, 1]
            y = -s*vertices[:, 0] + c*vertices[:, 1]
            vertices = numpy.array([x, y]).T

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
                self.fibre_size = config['ifu']['fibre_size']
            self.fibre_size *= astropy.units.micron

        self.padding = padding

        self.subifus = self._create_subifus()
        # list of lists of vertices
        self.polygon = [subifu.polygon for subifu in self.subifus]

        self.allow_rotation = allow_rotation

    def __repr__(self):

        return f'<IFU (name={self.name!r}, n_fibres={self.n_fibres}, centres={self.centres!s})>'

    @classmethod
    def from_config(cls):
        """Returns an `.IFU` object from the configuration file."""

        ifu_conf = config['ifu'].copy()

        name = ifu_conf.pop('type', None)

        return cls(name=name, **ifu_conf)

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

    def get_tile_grid(self, region, scale, tile_overlap=None, sparse=None, geodesic=None):
        """Returns a grid of positions that tile a region with this IFU.

        Parameters
        ----------
        region : ~lvmsurveysim.target.SkyRegion
            The SkyRegion to tile. It is assumed that x coordinates are RA
            and y is Declination, both in degrees.
        scale : float
            The scale in degrees per mm.
        tile_overlap : float
            The fraction of tile separation to overlap between neighboring tiles
            (ignored for sparse targets)
        sparse : float
            Factor for sparse sampling. Stretches IFU length scale by the number.
        geodesic : use geodesic sphere tiling, sparse gives depth in this case.

        """

        if isinstance(scale, astropy.units.Quantity):
            scale = scale.to('degree/mm').value

        points = []
        # Calculates the radius and apotheme of each subifu in degrees on the sky
        if sparse==None: 
            tile_overlap = tile_overlap or 0.0
        else:
            tile_overlap = 0.0

        sparse = sparse if sparse!=None else 1.0
        n_rows = self.subifus[0].n_rows
        ifu_phi_size = n_rows * self.fibre_size / 1000 * scale / 2. * sparse * (1.0-tile_overlap)
        ifu_theta_size = numpy.sqrt(3) / 2. * ifu_phi_size  # * (1.0-tile_overlap)

        # we are using an angular system theta, phi, with theta counted from the equator
        if geodesic == False:
            # Determine the centroid and bounds of the region
            centroid = numpy.array(region.centroid())
            minphi, mintheta, maxphi, maxtheta = region.bounds()

            # TODO: transform the Region object rather than the bounds!
            # Transform to a coordinate system where the region is on the equator. There we're natrually tiling
            # along great circles and the hexagon pattern is easiest to describe.
            Eq = EqTransform(centroid[0], 90.+centroid[1], 0.0)
            centroid = Eq.eq2gal(centroid[0], centroid[1])
            minphi, mintheta = Eq.eq2gal(minphi, mintheta)
            maxphi, maxtheta = Eq.eq2gal(maxphi, maxtheta)
            # transform might have caused issues with mod(360), so correct:
            if minphi >= maxphi:
                maxphi += 360.

            # The size of the grid in phi and theta, in degrees.
            size_phi  = numpy.abs(maxphi - minphi) * numpy.cos(numpy.radians(centroid[1]))
            size_theta = numpy.abs(maxtheta - mintheta)

            # The separation between grid points in RA and Dec
            delta_phi = 3 * ifu_phi_size
            delta_theta = ifu_theta_size

            # Calculates the initial positions of the grid points in RA and Dec.
            phi_pos = numpy.arange(-size_phi / 2., size_phi / 2. + delta_phi.value, delta_phi.value)
            theta_pos = numpy.arange(-size_theta / 2., size_theta / 2. + delta_theta.value, delta_theta.value)
            points = numpy.zeros((len(theta_pos), len(phi_pos), 2))

            # Offset each other row in phi by 1.5R
            points[:, :, 0] = phi_pos
            points[:, :, 0][1::2] += (1.5 * ifu_phi_size.value)
            # Set declination values
            points[:, :, 1] = theta_pos[numpy.newaxis].T

            # The separations in the phi axis must be converted to coordinate distances, but the cos(theta)
            # term cancels with the shortening of the length of the circle at theta
            points[:, :, 1] += centroid[1]
            points[:, :, 0] += centroid[0]

            # Reshape into a 2D list of points.
            points = points.reshape((-1, 2))

            # transform back to original coordinates and determine position angle
            points2 = numpy.array(Eq.gal2eq(points[:,0], points[:,1] + 1./3600.)).T
            points = numpy.array(Eq.gal2eq(points[:,0], points[:,1])).T
            pa = position_angle(numpy.deg2rad(points[:,0]), numpy.deg2rad(points[:,1]), 
                                numpy.deg2rad(points2[:,0]), numpy.deg2rad(points2[:,1]))
            pa = numpy.rad2deg(pa)
        else:
            x, y, z = lvmsurveysim.utils.geodesic_sphere.sphere(int(sparse))
            sk = astropy.coordinates.SkyCoord(x=x,y=y,z=z, representation_type='cartesian')
            sk.representation_type='spherical'
            points = numpy.zeros((len(sk),2))
            points[:,0] = sk.ra.deg
            points[:,1] = sk.dec.deg
            pa = numpy.zeros(len(sk))

        # Check what grid points would overlap with the region if occupied by an IFU.
        inside = [region.contains_point(x,y) for x,y in zip(points[:,0], points[:,1])]
        points_inside = points[inside]
        pa = pa[inside]
        return points_inside, pa

    def plot(self, show_fibres=False, fill=False):
        """Plots the IFU."""

        fig, ax = matplotlib.pyplot.subplots()

        for subifu in self.subifus:
            ax.add_patch(subifu.get_patch(fill=fill, edgecolor='r', linewidth=1, alpha=0.5))

            if show_fibres:
                ax.add_collection(subifu.get_patch_collection(ax))

        ax.autoscale_view()

        return fig
