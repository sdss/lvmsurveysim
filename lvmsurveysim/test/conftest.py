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

import pytest

import os
import pathlib
import yaml

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
                print(fn)
                fn.unlink()


@pytest.fixture
def plot(request):
    return request.config.getoption('--plot')


test_data = yaml.load(open(os.path.dirname(__file__) + '/test_data.yaml'))
test_regions = test_data['regions']


@pytest.fixture(params=test_regions.keys())
def region_type(request):
    """Yields the region type."""

    # This allows to limit the regions that are tested in a class by adding a
    # _region_type attribute to the class with the list of region types to test.
    if (hasattr(request.instance, '_region_types') and
            request.param not in request.instance._region_types):
        pytest.skip()
    else:
        yield request.param


@pytest.fixture()
def region(region_type):
    """Yields a `~lvmsurveysim.target.regions.Region`."""

    region_data = test_regions[region_type]

    yield Region(region_type, region_data['coords'], **region_data['params'])
