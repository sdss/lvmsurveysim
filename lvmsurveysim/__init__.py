#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-08
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-12 18:21:35

# flake8: noqa

import warnings

from .utils.config import get_config


config = get_config('~/.lvm/lvmsurveysim.yaml')


from .ifu import *
from .telescope import *


import astropy

warnings.filterwarnings('ignore',
                        message='Tried to get polar motions for times after IERS .*')

warnings.filterwarnings('ignore', category=astropy.utils.exceptions.ErfaWarning)


__version__ = '0.8.4'
