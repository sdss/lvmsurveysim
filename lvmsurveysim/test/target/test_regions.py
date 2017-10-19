#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 19, 2017
# @Filename: test_regions.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import shapely.geometry


class TestRegions(object):

    def test_shapely(self, region):

        assert hasattr(region, 'shapely')
        assert isinstance(region.shapely, shapely.geometry.base.BaseGeometry)
