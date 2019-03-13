#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2017-10-17
# @Filename: plot.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-12 18:59:46

import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy
import seaborn

from lvmsurveysim.target import _VALID_FRAMES


__all__ = ['get_axes', 'transform_patch_mollweide', 'convert_to_mollweide']


__MOLLWEIDE_ORIGIN__ = 120


def get_axes(projection='rectangular', frame='icrs'):
    """Returns axes for a particular projection.

    Parameters:
        projection ({'rectangular', 'mollweide'}):
            The type of projection of the axes returned. Either `rectangular`
            for a normal, cartesian, projection, or
            `mollweide <https://en.wikipedia.org/wiki/Mollweide_projection>`_.
        frame : str
            The reference frame of the axes. Must be one of
            `~lvmsurveysim.target.region._VALID_FRAMES`. Used to define
            the axis labels.

    Returns:
        fig, ax:
            The new matplotlib `~matplotlib.figure.Figure` and
            `~matplotlig.axes.Axes` objects for the selected projection.

    """

    assert frame in _VALID_FRAMES, 'invalid frame'

    with seaborn.axes_style('whitegrid'):

        if projection == 'rectangular':
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            ax.set_xlim(360, 0)
            ax.set_ylim(-20, 80)

        elif projection == 'mollweide':
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection='mollweide')
            org = __MOLLWEIDE_ORIGIN__

            tick_labels = numpy.array([150., 120, 90, 60, 30, 0,
                                       330, 300, 270, 240, 210])
            tick_labels = numpy.remainder(tick_labels + 360 + org, 360)
            tick_labels = numpy.array(tick_labels / 15., int)

            tickStr = []
            for tick_label in tick_labels[1::2]:
                tickStr.append('')
                tickStr.append('${0:d}^h$'.format(tick_label))

            ax.set_xticklabels(tickStr)  # we add the scale on the x axis
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


def convert_to_mollweide(coords):
    """Converts ``[0, 360)`` coordinates to Mollweide-valid values.

    Converts values to radians and offsets the Longitude to match the custom
    Mollweide projection used here.

    Parameters
    ----------
    coords : numpy.ndarray
        A ``Nx2`` array of coordinates to be converted.

    """

    coord0 = numpy.remainder(coords[:, 0] + 360 - __MOLLWEIDE_ORIGIN__, 360)
    ind = coord0 > 180.
    coord0[ind] -= 360
    coord0 = -coord0

    coords[:, 0] = coord0
    coords *= numpy.pi / 180.

    return coords


def transform_patch_mollweide(ax, patch, patch_centre=None):
    """Applies a transformation to the patch for the Mollweide projection.

    The Mollweide projection assumes the plotted values are in radians. In
    addition, the axes returned by `.get_axes` for a Mollweide projection have
    the tick labels modified to place the centre at a different position from
    the default 0 rad. This function applies a series of affine transformations
    to the input `~matplotlib.patches.Patch` to make the plot match the axes
    labels.

    Note that the Mollweide projection doesn't provide wrapping. Large regions
    that cross the edge of the projection will not be displayed completely.

    Parameters:
        ax (~matplotlib.axes.Axes):
            The axes on which the ``patch`` has been plotted.
        patch (`~matplotlib.patches.Patch`):
            The patch to be transformed.
        patch_centre (float):
            The RA value that will be used to determine the direction of the
            translation applied. If not defined, the best possible translation
            will be automatically determined.

    Returns:
        patch (`~matplotlib.patches.Patch`):
            The transformed patch.

    Example:

        Before calling `transform_patch_mollweide` the patch must have been
        added to the axes to ensure that the conversion between data and pixels
        is known ::

        >>> fig, ax = get_axes(projection='mollweide')
        >>> poly = Polygon([(0,0), (15,0), (15,15), (0,0)])
        >>> poly = ax.add_patch(poly)
        >>> poly_transformed = transform_patch_mollweide(ax, poly)

    """

    trans_to_rads = matplotlib.transforms.Affine2D().scale(numpy.pi / 180, numpy.pi / 180)
    trans_reflect = matplotlib.transforms.Affine2D(numpy.array([[-1, 0, 0],
                                                                [0, 1, 0],
                                                                [0, 0, 1]]))

    # If patch_centre is not defined, tries to figure out the centre from
    # the patch itself
    if patch_centre is None and hasattr(patch, 'center'):
        patch_centre = patch.center[0]

    # Calculates the best possible translation to match the tick labels.
    if patch_centre is None:
        translation = __MOLLWEIDE_ORIGIN__
    elif patch_centre < (-180 + __MOLLWEIDE_ORIGIN__) % 360:
        translation = __MOLLWEIDE_ORIGIN__
    else:
        translation = __MOLLWEIDE_ORIGIN__ + 360

    trans_origin = matplotlib.transforms.Affine2D().translate(numpy.radians(translation), 0)

    patch.set_transform(trans_to_rads + trans_reflect + trans_origin + ax.transData)

    return patch


def plot_ellipse(ax, ra, dec, width=3.0, height=None, org=0,
                 bgcolor='b', zorder=0, alpha=0.8):

    ra = numpy.atleast_1d(ra)
    dec = numpy.atleast_1d(dec)

    width = width or height

    ra = numpy.remainder(ra + 360 - org, 360)  # shift RA values
    ind = ra > 180.
    ra[ind] -= 360  # scale conversion to [-180, 180]
    ra = -ra        # reverse the scale: East to the left

    for ii in range(len(ra)):

        ell = matplotlib.pathces.Ellipse(
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
