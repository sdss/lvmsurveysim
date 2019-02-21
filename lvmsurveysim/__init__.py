#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-08
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-02-21 14:07:33

from .utils.config import get_config
from .utils.logger import log  # noqa


config = get_config('~/.lvm/lvmsurveysim.yaml')


__version__ = '0.1.0dev'
