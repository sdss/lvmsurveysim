#!/usr/bin/env python
# encoding: utf-8
#
# config.py
#
# Created by José Sánchez-Gallego on 18 Jun 2017.


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pathlib
import yaml


def merge(user, default):
    """Merges a user configuration with the default one."""

    if not user:
        return default

    if isinstance(user, dict) and isinstance(default, dict):
        for kk, vv in default.items():
            if kk not in user:
                user[kk] = vv
            else:
                user[kk] = merge(user[kk], vv)

    return user


def get_config(user_path):
    """Returns a dictionary object with configuration options."""

    user_path = pathlib.Path(user_path).expanduser()
    user = user_path.exists() and yaml.load(open(str(user_path), 'r'), Loader=yaml.FullLoader)

    default_path = pathlib.Path(__file__).parents[0] / '../etc/lvmsurveysim_defaults.yaml'
    default = yaml.load(open(str(default_path), 'r'), Loader=yaml.FullLoader)

    return merge(user, default)
