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
