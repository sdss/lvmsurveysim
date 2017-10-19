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
import yaml

from ..target.regions import Region


test_data = yaml.load(open(os.path.dirname(__file__) + '/test_data.yaml'))
test_regions = test_data['regions']


@pytest.fixture(scope='module', params=test_regions.keys())
def region(request):
    """Yields a `~lvmsurveysim.target.regions.Region`."""

    region_data = test_regions[request.param]

    yield Region(request.param, region_data['coords'], **region_data['params'])
