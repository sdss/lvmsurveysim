#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2017-10-17
# @Filename: region.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-13 11:32:39

import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.patches
import matplotlib.path
import matplotlib.transforms
import numpy
from copy import deepcopy

from spherical_geometry import polygon as sp
from lvmsurveysim.utils import plot as lvm_plot
from lvmsurveysim.exceptions import LVMSurveyOpsError, LVMSurveyOpsWarning

from . import _VALID_FRAMES


__all__ = ['SkyRegion']


# if we want to inherit: 
#super(SubClass, self).__init__('x')

class SkyRegion(object):
    """ This class represents a region on the sky.

    This class represents a region on the sky, parameterized either by 
    one of a number of common shapes, or by a set of vertices of a polygon.
    Internally all shapes are held as `~spherical_geometry.polygon` object
    so that the edges of the polygons are great circle segments on a sphere.

    The class provides convenience methods to construct such polygons using
    the Target parameterization from the target yaml file. It also provides
    methods to compute intersections between the regions and whether a point 
    is contained in the region (used later in tiling).

    Parameters:
    -----------
    typ : str
        String describing the shape, one of 'circle', 'ellipse', 'rectangle'
        'polygon' or 'raw'. Depending on the value of this parameter, we expect to find
        further parameters in **kwargs.
    coords : tuple of float
        Center coordinates for 'circle', 'ellipse', 'rectangle' regions,
        or tuple of vertices for 'polygon' in degrees.
        For 'raw', we expect the `~spherical_geometry.SphericalPolygon` object.
    **kwargs : dict
        Must contain keyword 'frame' set to 'icrs' or 'galactic'. 
        For 'rectangle' must contain 'width' and 'height' in degrees and 'pa' 
        a position angle (N through E) also in degrees.
        For 'circle' must contain 'r' with radius in degrees.
        For 'ellipse' must contain 'a', 'b', 'pa' with semi-axes and position angle
        in degrees.
        For 'raw' we expect only the 'frame' to be passed as a keyword argument.

    """

    def __init__(self, typ, coords, **kwargs):
        #print(typ, coords, kwargs)
        self.region_type = typ
        self.frame = kwargs['frame']

        if typ == 'rectangle':

            self.center = coords
            width = kwargs['width']
            height = kwargs['height']
            x0 = - width / 2.
            x1 = + width / 2.
            y0 = - height / 2.
            y1 = + height / 2.
            x, y = self._rotate_coords([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], kwargs['pa'])
            x, y = self._polygon_perimeter(x, y)
            x /= numpy.cos(numpy.deg2rad(y))
            y += self.center[1]
            x += self.center[0]
            self.region = sp.SphericalPolygon.from_radec(x, y, degrees=True)

        elif typ == 'circle':

            self.center = coords
            r = kwargs['r']
            k = int(numpy.max([numpy.floor(numpy.sqrt(r * 20)), 24]))
            x = numpy.array(list(reversed([r * numpy.cos(2.0*numpy.pi/k * i) for i in range(k+1)])))
            y = numpy.array(list(reversed([r * numpy.sin(2.0*numpy.pi/k * i) for i in range(k+1)])))
            y += self.center[1]
            x = x / numpy.cos(numpy.deg2rad(y)) + self.center[0]
            self.region = sp.SphericalPolygon.from_radec(x, y, center=self.center, degrees=True)
            # self.region = sp.SphericalPolygon.from_cone(coords[0], coords[1], kwargs['r'])
            # self.center = coords

        elif typ == 'ellipse':

            self.center = coords
            a, b = kwargs['a'], kwargs['b']
            k = int(numpy.max([numpy.floor(numpy.sqrt(((a + b) / 2) * 20)), 24]))
            x = list(reversed([a * numpy.cos(2.0*numpy.pi/k * i) for i in range(k+1)]))
            y = list(reversed([b * numpy.sin(2.0*numpy.pi/k * i) for i in range(k+1)]))
            x, y = self._rotate_coords(x, y, kwargs['pa'])
            y += self.center[1]
            x = x / numpy.cos(numpy.deg2rad(y)) + self.center[0]
            self.region = sp.SphericalPolygon.from_radec(x, y, center=self.center, degrees=True)

        elif typ == 'polygon':

            self.region_type = 'polygon'
            x, y = self._rotate_vertices(numpy.array(coords), 0.0)
            x, y = self._polygon_perimeter(x, y)
            self.center = [numpy.average(x), numpy.average(y)]
            x -= self.center[0]
            x = x / numpy.cos(numpy.deg2rad(y)) + self.center[0]
            self.region = sp.SphericalPolygon.from_radec(x, y, center=self.center, degrees=True)

        elif typ == 'raw':

            assert isinstance(coords, sp.SphericalPolygon), 'Raw SkyRegion reqiores SphericalPolygon.'
            self.region_type = 'polygon'
            self.region = coords
            self.frame = kwargs['frame']
            x, y = next(self.region.to_lonlat())
            self.center = [numpy.average(x), numpy.average(y)]

        else:
            raise LVMSurveyOpsError('Unknown region type '+typ)

    def __repr__(self):
        return f'<SkyRegion(type={self.region_type}, center={self.center}, frame={self.frame})>'

    def vertices(self):
        """ Return a `~numpy.array` of dimension Nx2 with the N vertices of the 
        SkyRegion.
        """
        i = self.region.to_lonlat()
        return numpy.array(next(i)).T        

    def bounds(self):
        """ Return a tuple of the bounds of the SkyRegion defined as the 
        minimum and maximum value of the coordinates in each dimension.
        """
        x, y = next(self.region.to_lonlat())
        return numpy.min(x), numpy.min(y), numpy.max(x), numpy.max(y)

    def centroid(self):
        """ Return the center coordinates of the SkyRegion
        """
        return self.center

    def intersects_poly(self, other):
        """ Return True if the SkyRegion intersects another SkyRegion.
        """
        assert self.frame == other.frame, "SkyRegions must be in the same coordinate frame for intersection."
        return self.region.intersects_poly(other.region)

    def contains_point(self, x, y):
        """ Return True if the point (x,y) is inside the region, 
        false otherwise.
        """
        return self.region.contains_lonlat(x, y, degrees=True)

    def icrs_region(self):
        """ Return a copy of the region transformed into the ICRS system.

        A deep-copy of the region with vertices transformed into the ICRS
        system is returned if the region is in any other reference frame.

        Returns
        -------
            `.SkyRegion` with vertices in the ICRS system.
 
        """
        r2 = deepcopy(self)
        if self.frame == 'icrs':
            return r2
        else:
            r2.frame = 'icrs'
            x, y = next(self.region.to_lonlat())
            c = SkyCoord(self.center[0]*u.deg, self.center[1]*u.deg).transform_to('icrs')
            s = SkyCoord(x*u.deg, y*u.deg, frame=self.frame).transform_to('icrs')
            r2.center = [c.ra.deg, c.dec.deg]
            r2.region = sp.SphericalPolygon.from_radec(s.ra.deg, s.dec.deg, degrees=True)
            return r2


    @classmethod
    def multi_union(cls, regions):
        """ Return a new SkyRegion of the union of the SkyRegions in `regions`.
        """
        def fchecker(f1, f2):
            assert f1==f2, "multi union must have uniform frame"
            return True
        
        frame = regions[0].frame
        [fchecker(frame, r.frame) for r in regions]
        return cls('raw', sp.SphericalPolygon.multi_union([r.region for r in regions]), frame=frame)


    @classmethod
    def _polygon_perimeter(cls, x, y, n=1.0, min_points=5):
        """ Subsample a polygon perimeter.
        
        This function returns new vertices along the perimeter of a polygon
        spaced `.n` degrees apart.

        Parameters
        ----------
        x, y : array-like
            x and y coordinates of the vertices of the original polygon.
        n : float
            optional, spacing for new vertices. defaults to 1.0
        min_points : int
            optional, minimal number of new points, defaults to 5

        Returns
        -------
        xp, yp : `~numpy.array`
            coordinates of new vertices
        """
        xp = numpy.array([])
        yp = numpy.array([])
        for x1,x2,y1,y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
            # Calculate the length of a segment, hopefully in degrees
            dl = ((x2-x1)**2 + (y2-y1)**2)**0.5
            n_dl = numpy.max([int(dl/n), min_points])
            
            if x1 != x2:
                m = (y2-y1)/(x2-x1)
                b = y2 - m*x2
                interp_x = numpy.linspace(x1, x2, num=n_dl, endpoint=False)
                interp_y = interp_x * m + b
            else:
                interp_x = numpy.full(n_dl, x1)
                interp_y = numpy.linspace(y1,y2, n_dl, endpoint=False)

            xp = numpy.append(xp, interp_x)
            yp = numpy.append(yp, interp_y)

        return xp, yp


    def plot(self, ax=None, projection='rectangular', return_patch=False, **kwargs):
        """Plots the region.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to use. If `None`, new axes will be created.
        projection (str):
            The projection to use. At this time, only ``rectangular`` and
            ``mollweide`` are accepted.
        return_patch (bool):
            If True, returns the
            `matplotlib patch <https://matplotlib.org/api/patches_api.html>`_
            for the region.
        kwargs (dict):
            Options to be passed to matplotlib when creating the patch.

        Returns
        -------
        Returns the matplotlib ~matplotlib.axes.Axes` object for this plot.
        If not specified, the default plotting styles will be used.
        If ``return_patch=True``, returns the patch as well.

        """

        if ax is None:
            fig, ax = lvm_plot.get_axes(projection=projection, frame=self.frame)

        coords = self.vertices()

        poly = matplotlib.path.Path(coords, closed=True)
        poly_patch = matplotlib.patches.PathPatch(poly, **kwargs)

        poly_patch = ax.add_patch(poly_patch)

        if projection == 'rectangular':
            #ax.set_aspect('equal', adjustable='box')

            min_x, min_y = coords.min(0)
            max_x, max_y = coords.max(0)

            padding_x = 0.1 * (max_x - min_x)
            padding_y = 0.1 * (max_y - min_y)

            ax.set_xlim(min_x - padding_x, max_x + padding_x)
            ax.set_ylim(min_y - padding_y, max_y + padding_y)

        elif projection == 'mollweide':

            poly_patch = lvm_plot.transform_patch_mollweide(ax, poly_patch, patch_centre=self.center[0])

        if return_patch:
            return fig, ax, poly_patch
        else:
            return fig, ax

    @classmethod
    def _rotate_vertices(cls, vertices, pa):
        sa, ca = numpy.sin(numpy.deg2rad(pa)), numpy.cos(numpy.deg2rad(pa))
        R = numpy.array([[ca, -sa], [sa, ca]])
        return numpy.dot(R, vertices.T).T 

    @classmethod
    def _rotate_coords(cls, x, y, pa):
        rot = cls._rotate_vertices(numpy.array([x, y]).T, pa)
        xyprime = rot.T
        return xyprime[0,:], xyprime[1,:]
