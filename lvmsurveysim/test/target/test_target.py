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

import pytest

from lvmsurveysim.target import Target
from lvmsurveysim.target import regions


test_targets = [('Target1', (169, 35), 'circle', {'r': 0.1})]


@pytest.mark.parametrize(('target_name', 'coords', 'target_type', 'target_params'), test_targets)
def test_targets(target_name, coords, target_type, target_params):

    target = Target(target_name, coords, target_type, target_params)
    assert isinstance(target, Target)

    assert hasattr(target, 'region')

    if target_type == 'circle':
        assert isinstance(target.region, regions.CircularRegion)
    elif target_type == 'ellipse':
        assert isinstance(target.region, regions.EllipticalRegion)
    elif target_type == 'polygon':
        assert isinstance(target.region, regions.PolygonalRegion)
    else:
        raise ValueError('invalid region type')
