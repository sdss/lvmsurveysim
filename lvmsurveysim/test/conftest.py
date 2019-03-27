#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 18, 2017
# @Filename: congtest.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pathlib
import yaml

import pytest

from ..target.regions import Region


def pytest_addoption(parser):
    parser.addoption('--plot', action='store_true', default=False, help='outputs test plots')


def pytest_configure(config):
    """Runs during configuration of conftest."""

    do_plot = config.getoption('--plot')

    if do_plot:
        plots_path = (pathlib.Path(__file__).parent / 'plots')
        if plots_path.exists():
            for fn in plots_path.glob('*'):
                fn.unlink()


@pytest.fixture
def plot(request):
    return request.config.getoption('--plot')


test_data = yaml.load(open(os.path.dirname(__file__) + '/test_data/test_data.yaml'))
test_regions = test_data['regions']


@pytest.fixture(params=test_regions.keys())
def region_name(request):
    """Yields the parameters for a region."""

    # This allows to limit the regions that are tested in a class by adding a
    # _region_type attribute to the class with the list of region types to test.
    # We may have multiple regions of the same type (e.g., elliptical,
    # elliptical_2, elliptical_oblong). We want to accept all the ones that
    # start with a value in _region_types
    if (not hasattr(request.instance, '_region_type') or
            test_regions[request.param]['type'] == request.instance._region_type):
        yield request.param
    else:
        pytest.skip()


@pytest.fixture()
def region(region_name):
    """Yields a `~lvmsurveysim.target.regions.Region`."""

    region_data = test_regions[region_name]

    params = region_data['params'] if 'params' in region_data else {}

    region = Region(region_data['type'], region_data['coords'], **params)
    region.name = region_name  # this can be useful to differentiate multiple regions of same type.

    yield region


@pytest.fixture()
def test_target_file():
    yield pathlib.Path(__file__).parent / 'test_data/test_targets.yaml'
