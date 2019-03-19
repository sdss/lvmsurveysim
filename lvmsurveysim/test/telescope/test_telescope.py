#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Oct 27, 2017
# @Filename: test_telescope.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest

from lvmsurveysim.telescope import Telescope


@pytest.mark.parametrize(('name', 'diameter', 'f'), [('my_telescope', 1, 5),
                                                     ('APO-1m', None, None)])
def test_telescope(name, diameter, f):

    teles = Telescope(name, diameter, f)

    assert teles.name == name

    if diameter is not None:
        assert teles.diameter.value == diameter

    if f is not None:
        assert teles.f == f

    assert teles.focal_length.value == 5
    assert teles.plate_scale.value == pytest.approx(41.253)


def test_telescope_fails_configuration():

    with pytest.raises(AssertionError):
        Telescope('a_bad_telescope_name')
