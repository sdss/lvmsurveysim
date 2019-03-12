#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-08
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-11 22:36:05

# flake8: noqa

from .utils.config import get_config
from .utils.logger import log


config = get_config('~/.lvm/lvmsurveysim.yaml')


from .ifu import *
from .telescope import *


__version__ = '0.1.0dev'
