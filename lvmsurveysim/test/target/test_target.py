#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 26, 2017
# @Filename: test_target.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import pathlib

import pytest

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import lvmsurveysim
from lvmsurveysim.target import Target
from lvmsurveysim.target import regions


@pytest.fixture()
def restore_config():

    config_copy = copy.deepcopy(lvmsurveysim.config)
    yield
    lvmsurveysim.config = config_copy


test_targets = [('Target1', (169, 35), 'circle', {'r': 0.1})]


@pytest.mark.parametrize(('target_name', 'coords', 'target_type', 'target_params'), test_targets)
def test_targets(target_name, coords, target_type, target_params):

    target = Target(target_name, coords, target_type, target_params)
    assert isinstance(target, Target)

    assert target.name == target_name

    assert hasattr(target, 'region')

    if target_type == 'circle':
        assert isinstance(target.region, regions.CircularRegion)
    elif target_type == 'ellipse':
        assert isinstance(target.region, regions.EllipticalRegion)
    elif target_type == 'polygon':
        assert isinstance(target.region, regions.PolygonalRegion)
    else:
        raise ValueError('invalid region type')


def test_target_from_list(restore_config):

    test_target_list = pathlib.Path(__file__).parents[1] / 'test_data/test_targets.yaml'

    assert test_target_list.exists()

    lvmsurveysim.config['target_list'] = str(test_target_list)

    target_1 = Target.from_target_list('Target1')

    assert target_1.name == 'Target1'
    assert isinstance(target_1, Target)
    assert isinstance(target_1.region, regions.EllipticalRegion)


def test_plot_target_mollweide(plot):

    target = Target('MyTarget', [(100, 65), (120, 70), (110, 70)], region_type='polygon')

    assert isinstance(target.region, regions.PolygonalRegion)

    fig, ax = target.plot(projection='mollweide')

    if plot:
        plot_path = pathlib.Path(__file__).parents[1] / 'plots/test_mytarget_mollweide.pdf'
        plt.savefig(str(plot_path))

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, Axes)
