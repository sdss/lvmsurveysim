#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 27, 2017
# @Filename: test_tiling.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from lvmsurveysim.target import Target
from lvmsurveysim.telescope import Telescope
from lvmsurveysim.tiling import Tiling


def test_tiling_no_ifu(test_target_file):

    target_1 = Target.from_target_list('Target1', target_list=test_target_file)
    telescope = Telescope('APO-1m')

    assert isinstance(target_1, Target)

    tiling = Tiling(target_1, telescope)

    assert isinstance(tiling, Tiling)
    assert len(tiling.tiles) == 0
