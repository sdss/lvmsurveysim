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

import astropy.coordinates
import astropy.units
import matplotlib.patches
import matplotlib.path
import matplotlib.transforms
import numpy

from spherical_geometry import polygon as sp
from lvmsurveysim.utils import plot as lvm_plot
from lvmsurveysim.exceptions import LVMSurveyOpsError, LVMSurveyOpsWarning

from . import _VALID_FRAMES


__all__ = ['SkyRegion']


# if we want to inherit: 
#super(SubClass, self).__init__('x')

class SkyRegion(object):
    frame = 'icrs'
    def __init__(self, typ, coords, **kwargs):
        print(typ, coords, kwargs)
        self.typ = typ
        if typ == 'circle':

            self.region = sp.SphericalPolygon.from_cone(coords[0], coords[1], kwargs['r'])

        elif typ == 'rectangle':

            width_deg = kwargs['width']
            height_deg = kwargs['height']
            x0 = coords[0] - width_deg / 2.
            x1 = coords[0] + width_deg / 2.
            y0 = coords[1] - height_deg / 2.
            y1 = coords[1] + height_deg / 2.
            x, y = self._rotate_coords([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], kwargs['pa'])
            x = x/numpy.cos(numpy.deg2rad(y))
            self.region =  sp.SphericalPolygon.from_radec(x, y, center=coords, degrees=True)

        elif typ == 'ellipse':

            a, b = kwargs['a'], kwargs['b']
            k = numpy.max(numpy.floor(numpy.sqrt(((a + b) / 2) * 20)), 24)
            x = [coords[0] + a * numpy.cos(2.0*numpy.pi/k * i) for i in range(k+1)]
            y = [coords[1] + b * numpy.cos(2.0*numpy.pi/k * i) for i in range(k+1)]
            x, y = self._rotate_coords(x, y, kwargs['pa'])
            x = x/numpy.cos(numpy.deg2rad(y))
            self.region =  sp.SphericalPolygon.from_radec(x, y, center=coords, degrees=True)

        elif typ == 'polygon':

            x, y = self._rotate_vertices(numpy.array(coords), kwargs['pa'])
            x = x/numpy.cos(numpy.deg2rad(y))
            self.region =  sp.SphericalPolygon.from_radec(x, y, degrees=True)

        else:
            raise LVMSurveyOpsError('Unknown region type '+typ)

    def vertices(self):
        i = self.region.to_lonlat()
        return numpy.array(next(i)).T


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
            __, ax = lvm_plot.get_axes(projection=projection, frame=self.frame)

        coords = self.vertices()

        poly = matplotlib.path.Path(coords, closed=True)
        poly_patch = matplotlib.patches.PathPatch(poly, **kwargs)

        poly_patch = ax.add_patch(poly_patch)

        if projection == 'rectangular':
            ax.set_aspect('equal', adjustable='box')

            min_x, min_y = coords.min(0)
            max_x, max_y = coords.max(0)

            padding_x = 0.1 * (max_x - min_x)
            padding_y = 0.1 * (max_y - min_y)

            ax.set_xlim(min_x - padding_x, max_x + padding_x)
            ax.set_ylim(min_y - padding_y, max_y + padding_y)

        elif projection == 'mollweide':

            centroid = self.region.inside()
            poly_patch = lvm_plot.transform_patch_mollweide(ax, poly_patch,
                                                            patch_centre=centroid)

        if return_patch:
            return ax, poly_patch
        else:
            return ax

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

# def rotate_vertices(vertices, pa):
#     sa, ca = numpy.sin(numpy.deg2rad(pa)), numpy.cos(numpy.deg2rad(pa))
#     R = numpy.array([[ca, -sa], [sa, ca]])
#     return numpy.dot(R, vertices.T).T 

# def rotate_coords(x, y, pa):
#     rot = rotate_vertices(numpy.array([x, y]).T, pa)
#     xyprime = rot.T
#     return xyprime[0,:], xyprime[1,:]

# def plot(vertices, **kwargs):
#     fig, ax = lvm_plot.get_axes(projection='rectangular', frame='icrs')

#     min_x, min_y = vertices.min(0)
#     max_x, max_y = vertices.max(0)

#     padding_x = 0.1 * (max_x - min_x)
#     padding_y = 0.1 * (max_y - min_y)

#     ax.set_xlim(min_x - padding_x, max_x + padding_x)
#     ax.set_ylim(min_y - padding_y, max_y + padding_y)

#     poly = matplotlib.path.Path(vertices, closed=True)
#     poly_patch = matplotlib.patches.PathPatch(poly, **kwargs)
#     poly_patch = ax.add_patch(poly_patch)

#     ax.set_aspect('equal', adjustable='box')

#     return fig, ax
