#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2017-10-17
# @Filename: plot.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-29 01:23:34

import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy
import seaborn

from lvmsurveysim.target import _VALID_FRAMES


__all__ = ['get_axes', 'transform_patch_mollweide', 'convert_to_mollweide', 'plot_ellipse']


__MOLLWEIDE_ORIGIN__ = 180


def get_axes(projection='rectangular', frame='icrs', ylim=None):
    """Returns axes for a particular projection.

    Parameters
    ----------
    projection : str
        The type of projection of the axes returned. Either ``'rectangular'``
        for a normal, cartesian, projection, or
        `mollweide <https://en.wikipedia.org/wiki/Mollweide_projection>`_.
    frame : str
        The reference frame of the axes. Must be one of
        `~lvmsurveysim.target._VALID_FRAMES`. Used to define
        the axis labels.
    ylim : tuple
        The range to be used to limit the y-axis. Only relevant if
        ``projection='rectangular'``. If `None`, ``(-90, 90)`` will be used.

    Returns
    -------
    figax
        The new matplotlib `~matplotlib.figure.Figure` and
        `~matplotlig.axes.Axes` objects for the selected projection.

    """

    assert frame in _VALID_FRAMES, 'invalid frame'

    with seaborn.axes_style('whitegrid'):

        if projection == 'rectangular':
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            ax.set_xlim(360, 0)

            if ylim:
                ax.set_ylim(**ylim)
            else:
                ax.set_ylim(-90, 90)

        elif projection == 'mollweide':
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection='mollweide')
            org = __MOLLWEIDE_ORIGIN__

            tick_labels = numpy.array([150., 120, 90, 60, 30, 0,
                                       330, 300, 270, 240, 210])
            tick_labels = numpy.remainder(tick_labels + 360 + org, 360)
            tick_labels = numpy.array(tick_labels / 15., int)

            tickStr = []
            for tick_label in tick_labels:
                #tickStr.append('')
                tickStr.append('${0:d}^h$'.format(tick_label))

            # Bug fix: if we have an even number of ticklabels, starting at 1 and skipping ever other will produce a mismatch in the number of tics and lables. 
            # Temporary fix, try to set them, but don't break if the mismatch exists.
            try:
                ax.set_xticklabels(tickStr)  # we add the scale on the x axis
            except:
                pass
            ax.grid(True)

        else:
            raise ValueError('invalid projection')

        if frame == 'icrs':
            ax.set_xlabel(r'$\alpha_{2000}\,{\rm [deg]}$')
            ax.set_ylabel(r'$\delta_{2000}\,{\rm [deg]}$')
        elif frame == 'galactic':
            ax.set_xlabel(r'$\rm l\,[deg]$')
            ax.set_ylabel(r'$\rm b\,[deg]$')

    return fig, ax


def convert_to_mollweide(ra, dec):
    """Converts ``[0, 360)`` coordinates to Mollweide-valid values.

    Converts values to radians and offsets the Longitude to match the custom
    Mollweide projection used here. Flips RA so that E is left.

    Parameters
    ----------
    ra : ~numpy.ndarray
        ra in degrees of coordinates to be converted
    dec : ~numpy.ndarray
        dec in degrees of coordinates to be converted

    Returns
    -------
    ra0,dec0 in radians suitable to plot in mollwede projection
    """

    ra = numpy.asarray([ra]) if numpy.isscalar(ra) else numpy.asarray(ra)
    dec = numpy.asarray([dec]) if numpy.isscalar(dec) else numpy.asarray(dec)
    ra0 = numpy.remainder(ra + 360 - __MOLLWEIDE_ORIGIN__, 360) # shift RA values
    ra0[ra0 > 180.] -= 360                                      # convert range to [-180,180]
    return -numpy.deg2rad(ra0),numpy.deg2rad(dec)               # reverse RA so that East is left


def transform_vertices_mollweide(vertices):
    """Applies a transformation to a set of vertices for the Mollweide projection.

    The Mollweide projection assumes the plotted values are in radians. In
    addition, the axes returned by `.get_axes` for a Mollweide projection have
    the tick labels modified to place the centre at a different position from
    the default 0 rad. RA is flipped, so that East is left.

    See also `.transform_to_mollweide`.

    Note that the Mollweide projection doesn't provide wrapping. Large regions
    that cross the edge of the projection will not be displayed completely.

    Parameters:
        vertices : `~numpy.array`
            The Nx2 array of vertices to be transformed.

    Returns:
        vertices : `~numpy.array`
            The Nx2 array of vertices in Mollweide coordinates.

    """
    v = vertices.T
    r, d = convert_to_mollweide(v[0,:], v[1,:])
    return numpy.array([r,d]).T



def transform_patch_mollweide(patch):
    """Applies a transformation to a `~matplotlib.patch` for the Mollweide projection.

    The Mollweide projection assumes the plotted values are in radians. In
    addition, the axes returned by `.get_axes` for a Mollweide projection have
    the tick labels modified to place the centre at a different position from
    the default 0 rad. RA is flipped, so that East is left.

    See also `.transform_to_mollweide`.

    Note that the Mollweide projection doesn't provide wrapping. Large regions
    that cross the edge of the projection will not be displayed completely.

    Parameters:
        patch : `~matplotlib.patch`
            The patch to be transformed.

    Returns:
        patch : `~matplotlib.patch`
            The transformed patch.
    
    """
    v = patch.get_xy().T
    r, d = convert_to_mollweide(v[0,:], v[1,:])
    patch.set_xy(numpy.array([r,d]).T)
    return patch


def plot_ellipse(ax, ra, dec, width=3.0, height=None, origin=0,
                 bgcolor='b', zorder=0, alpha=0.8):
    """Plots an ellipse path of a given angular size."""

    ra = numpy.atleast_1d(ra)
    dec = numpy.atleast_1d(dec)

    height = width or height

    ra = numpy.remainder(ra + 360 - origin, 360)  # shift RA values
    ind = ra > 180.
    ra[ind] -= 360  # scale conversion to [-180, 180]
    ra = -ra        # reverse the scale: East to the left

    for ii in range(len(ra)):

        ell = matplotlib.patches.Ellipse(
            xy=(numpy.radians(ra[ii]), numpy.radians(dec[ii])),
            width=numpy.radians(width) / numpy.cos(numpy.radians(dec[ii])),
            height=numpy.radians(height),
            edgecolor='None',
            lw=0.0,
            facecolor=bgcolor,
            zorder=zorder,
            alpha=alpha)

        ax.add_patch(ell)

    return ax
