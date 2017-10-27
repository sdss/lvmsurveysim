#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 27, 2017
# @Filename: test_ifu.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest

import numpy.testing

import lvmsurveysim
from lvmsurveysim.tiling import ifu


ifu_data = lvmsurveysim.config['ifus']


@pytest.mark.parametrize(('ifu', 'ifu_name'), [(ifu.MonolithicIFU(), 'monolithic'),
                                               (ifu.AbuttableTriangleIFU(), 'abuttable_triangle')])
def test_ifus(ifu, ifu_name):

    config_data = ifu_data[ifu_name]

    assert len(ifu.subifus) == len(config_data['centres'])

    numpy.testing.assert_almost_equal(ifu.centres, config_data['centres'])

    assert ifu.n_fibres == config_data['n_fibres']
    assert ifu.n_ifus == config_data['n_ifus']
