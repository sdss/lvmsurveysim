#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-07-08
# @Filename: __init__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-03-11 18:02:18

# flake8: noqa

from astropy import log

try:
    log.disable_warnings_logging()
    log.disable_exception_logging()
except:
    pass


from .utils.config import get_config
from .utils.logger import log


config = get_config('~/.lvm/lvmsurveysim.yaml')


from .ifu import *
from .telescope import *


__version__ = '0.1.0dev'
