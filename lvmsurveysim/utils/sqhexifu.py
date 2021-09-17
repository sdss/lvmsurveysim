#!/usr/bin/env python
# encoding: utf-8
#
# ifu.py
#
# Created by José Sánchez-Gallego on 5 Sep 2017.


from __future__ import absolute_import, division, print_function

import matplotlib.patches
import matplotlib.pyplot
import numpy
import astropy.units
from astropy.coordinates.angle_utilities import position_angle


import lvmsurveysim
from lvmsurveysim import config
from lvmsurveysim.utils import plot as lvm_plot
from lvmsurveysim.utils.plot import __MOLLWEIDE_ORIGIN__, get_axes, transform_patch_mollweide, convert_to_mollweide


__all__ = ['SQIFU', 'EqTransform']


def test(ra0, dec0, l0):
    Eq = EqTransform(ra0, dec0+90., l0)
    print(Eq.eq2gal(ra0, dec0), Eq.gal2eq(0, 0))

def test2():
    # s=SQIFU('square', Region([245.,-43.,], 25, 1))
    # f,a = s.plot_tiling()
    # f.show()

    s=SQIFU('hex', Region([245.,-43.,], 10, 10))
    f,a = s.plot_tiling()
    f.show()


class EqTransform(object):
    '''
    The transformation between the equatorial and galactic systems consisted of:
    1. a rotation around the celestial polar axis by 282.25 deg, so that the reference zero longitude matches the node
    2. a rotation around the node by 62.9 deg
    3. a rotation around the galactic polar axis by 33 deg so that the zero longitude meridian matches the galactic center.
    '''
    def __init__(self, RA_NGP, DEC_NGP, L_CP):
        self.RA_NGP = numpy.deg2rad(RA_NGP)   # numpy.deg2rad(192.8594812065348)
        self.DEC_NGP = numpy.deg2rad(DEC_NGP) # numpy.deg2rad(27.12825118085622)
        self.L_CP = numpy.deg2rad(L_CP)       # numpy.deg2rad(122.9319185680026)

        self.L_0 = self.L_CP - numpy.pi / 2.
        self.RA_0 = self.RA_NGP + numpy.pi / 2.
        self.DEC_0 = numpy.pi / 2. - self.DEC_NGP

    def gal2eq(self, l, b):
        l, b = numpy.deg2rad(numpy.array(l, ndmin=1)), numpy.deg2rad(numpy.array(b, ndmin=1))
        sind = numpy.sin(b) * numpy.sin(self.DEC_NGP) + numpy.cos(b) * numpy.cos(self.DEC_NGP) * numpy.sin(l - self.L_0)
        dec = numpy.arcsin(sind)
        cosa = numpy.cos(l - self.L_0) * numpy.cos(b) / numpy.cos(dec)
        sina = (numpy.cos(b) * numpy.sin(self.DEC_NGP) * numpy.sin(l - self.L_0) - numpy.sin(b) * numpy.cos(self.DEC_NGP)) / numpy.cos(dec)
        return self.normalize_angles(cosa, sina, self.RA_0, dec)

    def eq2gal(self, ra, dec):
        ra, dec = numpy.deg2rad(numpy.array(ra, ndmin=1)), numpy.deg2rad(numpy.array(dec, ndmin=1))
        numpy.sinb = numpy.sin(dec) * numpy.cos(self.DEC_0) - numpy.cos(dec) * numpy.sin(ra - self.RA_0) * numpy.sin(self.DEC_0)
        b = numpy.arcsin(numpy.sinb)
        cosl = numpy.cos(dec) * numpy.cos(ra - self.RA_0) / numpy.cos(b)
        sinl = (numpy.sin(dec) * numpy.sin(self.DEC_0) + numpy.cos(dec) * numpy.sin(ra - self.RA_0) * numpy.cos(self.DEC_0)) / numpy.cos(b)
        return self.normalize_angles(cosl, sinl, self.L_0, b)

    def normalize_angles(self, cosa, sina, ra0, dec):
        dec = numpy.rad2deg(dec)
        cosa[cosa < -1.0] = -1.0
        cosa[cosa > 1.0] = 1.0
        ra = numpy.arccos(cosa)
        ra[numpy.where(sina < 0.)] *= -1.0
        ra = numpy.rad2deg(ra + ra0)
        ra = numpy.mod(ra, 360.)
        dec = numpy.mod(dec + 90., 180.) - 90.
        return ra, dec


class Region(object):
    def __init__(self, center, sizex, sizey):
        self.center = center
        self.extent = [center[0]-sizex, center[1]-sizey, center[0]+sizex, center[1]+sizey]

    def centroid(self):
        return self.center

    def bounds(self):
        return self.extent
    
    def contains_point(self, x,y):
        return (x>=self.extent[0]) and (y>=self.extent[1]) and (x<self.extent[2]) and (y<self.extent[3])


class SQIFU(object):

    def __init__(self, typ, region):

        self.typ = typ
        self.center = [0.0, 0.0]
        self.scale = 1.0  #degrees/IFU (mm)
        self.region = region
        self.tiles = None

        if typ=='square':
            self.polygon = self._create_sq_polygon()
        else:
            self.polygon = self._create_hex_polygon()


    def _create_hex_polygon(self):
        """Creates a list of polygon vertices representing the shape of this sub-IFU."""

        RR = 0.5                   # Assumes unitary diameter
        rr = numpy.sqrt(3) / 2. * RR  # Inner radius
        cos60 = 0.5

        xx, yy = self.center
        vertices = [(xx - RR, yy),
                    (xx - RR * cos60, yy + rr),
                    (xx + RR * cos60, yy + rr),
                    (xx + RR, yy),
                    (xx + RR * cos60, yy - rr),
                    (xx - RR * cos60, yy - rr)]
        
        return vertices

    def _create_sq_polygon(self):
        """Creates a list of polygon vertices representing the shape of this sub-IFU."""

        RR = 0.5
        vertices = [(-RR, -RR),
                    (RR, -RR),
                    (RR, RR),
                    (-RR, RR)]
        
        return vertices


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
            rr_deg = self.scale / 2.
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


    def get_tile_grid(self, region, scale, tile_overlap=None, sparse=None, geodesic=None):

        if isinstance(scale, astropy.units.Quantity):
            scale = scale.to('degree/mm').value

        points = []
        # Calculates the radius and apotheme of each subifu in degrees on the sky
        if sparse==None: 
            tile_overlap = tile_overlap or 0.0
        else:
            tile_overlap = 0.0

        sparse = sparse if sparse!=None else 1.0

        if self.typ =='hex':
            scale = scale / 2 
        ifu_phi_size = 1.0 * scale * sparse * (1.0-tile_overlap)
        if self.typ == 'square':
            ifu_theta_size = ifu_phi_size
        else:
            ifu_theta_size = numpy.sqrt(3) / 2. * ifu_phi_size

        # we are using an angular system theta, phi, with theta counted from the equator
        if geodesic == False:
            # Determine the centroid and bounds of the region, this is in whatever coordinate frame
            # the region is in
            centroid = numpy.array(region.centroid())
            minphi, mintheta, maxphi, maxtheta = region.bounds()
            Eq = EqTransform(centroid[0], 90.+centroid[1], 0.0)
            centroid = Eq.eq2gal(centroid[0], centroid[1])

            minphi, mintheta = Eq.eq2gal(minphi, mintheta)
            maxphi, maxtheta = Eq.eq2gal(maxphi, maxtheta)
            # The size of the grid in phi and theta, in degrees.
            size_phi  = numpy.abs(maxphi - minphi) * numpy.cos(numpy.radians(centroid[1]))
            size_theta = numpy.abs(maxtheta - mintheta)
            # The separation between grid points in phi and theta
            if self.typ == 'square':
                delta_phi = ifu_phi_size
            else:
                delta_phi = 3 * ifu_phi_size

            delta_theta = ifu_theta_size

            # Calculates the initial positions of the grid points in RA and Dec.
            phi_pos = numpy.arange(-size_phi / 2., size_phi / 2. + delta_phi, delta_phi)
            theta_pos = numpy.arange(-size_theta / 2., size_theta / 2. + delta_theta, delta_theta)
            points = numpy.zeros((len(theta_pos), len(phi_pos), 2))

            points[:, :, 0] = phi_pos
            points[:, :, 1] = theta_pos[numpy.newaxis].T

            if self.typ=='hex':
                points[:, :, 0][1::2] += (1.5 * ifu_phi_size)

            # The separations in the phi axis must be converted to coordinate 
            # distances in RA/l using the local DEC or b on the sky
            points[:, :, 1] += centroid[1]
            #points[:, :, 0] /= numpy.cos(numpy.radians(points[:, :, 1]))
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

        return points[inside], pa[inside]


    def tile(self, to_frame=None):

        coords, pa = self.get_tile_grid(self.region, self.scale, 
                                        tile_overlap=0.0, sparse=None, geodesic=False)
        tiles = astropy.coordinates.SkyCoord(coords[:, 0], coords[:, 1], frame='icrs', unit='deg')
        # second set offset in dec to find position angle after transform
        tiles2 = astropy.coordinates.SkyCoord(coords[:, 0], coords[:, 1]+1./3600, frame='icrs', unit='deg')

        # transform not only centers, but also second set of coordinates slightly north, then compute the angle
        if to_frame:
            tiles = tiles.transform_to(to_frame)
            tiles2 = tiles2.transform_to(to_frame)
        self.pa = pa + tiles.position_angle(tiles2)

        # cache the new tiles and the priorities
        self.tiles = tiles


    def plot(self, fill=False):
        """Plots the IFU."""

        fig, ax = matplotlib.pyplot.subplots()

        ax.add_patch(self.get_patch(fill=fill, edgecolor='r', linewidth=1, alpha=0.5))

        ax.autoscale_view()

        return fig


    def plot_tiling(self, projection='rectangular', fig=None, **kwargs):

        if self.tiles is None:
            self.tile(to_frame='icrs')

        lon, lat = self.tiles.ra.deg, self.tiles.dec.deg

        if fig is None:
            fig, ax = lvm_plot.get_axes(projection=projection, frame='icrs')
        else:
            ax = fig.axes[0]

        if projection=='mollweide':
            c1,c2 = lvm_plot.convert_to_mollweide(lon, lat)
        else:
            c1, c2 = lon, lat

        patches = [self.get_patch(scale=self.scale, centre=[c1[p], c2[p]], pa=self.pa[p],
                                      edgecolor='r', linewidth=1, alpha=0.5)
                    for p in range(len(c1))]

        if projection == 'mollweide':
            patches = [transform_patch_mollweide(patch) for patch in patches]

        for patch in patches:
            ax.add_patch(patch)

        ax.scatter(c1, c2, s=1, **kwargs)

        return fig, ax
