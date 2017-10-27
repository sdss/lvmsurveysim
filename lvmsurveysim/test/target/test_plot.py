#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 26, 2017
# @Filename: test_plot.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pathlib

import pytest

import matplotlib.pyplot as plt
from matplotlib import patches

import lvmsurveysim.target.plot


poly_0_0 = patches.Polygon([(0, 0), (15, 0), (15, 15), (0, 0)])
poly_120_0 = patches.Polygon([(120, 0), (135, 0), (135, 15), (120, 0)])
poly_180_0 = patches.Polygon([(180, 0), (195, 0), (195, 15), (180, 0)])
poly_270_0 = patches.Polygon([(270, 0), (285, 0), (285, 15), (270, 0)])
poly_345_0 = patches.Polygon([(345, 0), (360, 0), (360, 15), (345, 0)])


@pytest.mark.parametrize(('name', 'patch', 'patch_centre'), [('0_0', poly_0_0, 0),
                                                             ('120_0', poly_120_0, 120),
                                                             ('180_0', poly_180_0, 180),
                                                             ('270_0', poly_270_0, 270),
                                                             ('345_0', poly_345_0, 345)])
def test_mollweide(name, patch, patch_centre, plot):

    fig, ax = lvmsurveysim.target.plot.get_axes(projection='mollweide')
    patch = ax.add_patch(patch)
    lvmsurveysim.target.plot.transform_patch_mollweide(ax, patch, patch_centre=patch_centre)

    if plot:
        plot_path = pathlib.Path(__file__).parents[1] / f'plots/test_plot_{name}_mollweide.pdf'
        plt.savefig(str(plot_path))


def test_mollweide_no_patch_centre(plot):

    fig, ax = lvmsurveysim.target.plot.get_axes(projection='mollweide')
    patch = patches.Ellipse((345, 0), width=20, height=15)
    patch = ax.add_patch(patch)

    lvmsurveysim.target.plot.transform_patch_mollweide(ax, patch, patch_centre=None)

    if plot:
        plot_path = pathlib.Path(__file__).parents[1] / f'plots/test_plot_no_patch_centre.pdf'
        plt.savefig(str(plot_path))
