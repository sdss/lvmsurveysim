#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 17, 2017
# @Filename: plot.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import matplotlib.transforms

import numpy as np

import seaborn as sns


__all__ = ['get_axes', 'transform_patch_mollweide']


__MOLLWEIDE_ORIGIN__ = 120


def get_axes(projection='rectangular'):
    """Returns axes for a particular projection.

    Parameters:
        projection ({'rectangular', 'mollweide'}):
            The type of projection of the axes returned. Either `rectangular`
            for a normal, cartesian, projection, or
            `mollweide <https://en.wikipedia.org/wiki/Mollweide_projection>`_.

    Returns:
        fig, ax:
            The new matplotlib `~matplotlib.figure.Figure` and
            `~matplotlig.axes.Axes` objects for the selected projection.

    """

    with sns.axes_style('whitegrid'):

        if projection == 'rectangular':
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            ax.set_xlabel(r'$\alpha_{2000}$')
            ax.set_ylabel(r'$\delta_{2000}$')

            ax.set_xlim(360, 0)
            ax.set_ylim(-20, 80)

        elif projection == 'mollweide':
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection='mollweide')
            org = __MOLLWEIDE_ORIGIN__

            tick_labels = np.array([150., 120, 90, 60, 30, 0,
                                    330, 300, 270, 240, 210])
            tick_labels = np.remainder(tick_labels + 360 + org, 360)
            tick_labels = np.array(tick_labels / 15., int)

            tickStr = []
            for tick_label in tick_labels[1::2]:
                tickStr.append('')
                tickStr.append('${0:d}^h$'.format(tick_label))

            ax.set_xticklabels(tickStr)  # we add the scale on the x axis
            ax.grid(True)

            ax.set_xlabel(r'$\alpha_{2000}$')
            ax.set_ylabel(r'$\delta_{2000}$')

        else:
            raise ValueError('invalid projection')

    return fig, ax


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

    trans_to_rads = matplotlib.transforms.Affine2D().scale(np.pi / 180, np.pi / 180)
    trans_reflect = matplotlib.transforms.Affine2D(np.array([[-1, 0, 0],
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

    trans_origin = matplotlib.transforms.Affine2D().translate(np.radians(translation), 0)

    patch.set_transform(trans_to_rads + trans_reflect + trans_origin + ax.transData)

    return patch
