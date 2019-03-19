#!/usr/bin/env python
# encoding: utf-8
#
# @Author: José Sánchez-Gallego
# @Date: Aug 15, 2017
# @Filename: setup.py
# @License: BSD 3-Clause
# @Copyright: José Sánchez-Gallego


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from setuptools import setup, find_packages

import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_requires = [line.strip().replace('==', '>=') for line in open(requirements_file)
                    if not line.strip().startswith('#') and line.strip() != '']

NAME = 'lvmsurveysim'
# do not use x.x.x-dev.  things complain.  instead use x.x.xdev
VERSION = '0.1.1dev'
RELEASE = 'dev' not in VERSION


def run():

    setup(name=NAME,
          version=VERSION,
          license='BSD3',
          description='Survey simulations and tiling for LVM',
          long_description=open('README.rst').read(),
          author='José Sánchez-Gallego',
          author_email='gallegoj@uw.edu',
          keywords='LVM simulation survey scheduling',
          url='https://github.com/sdss/lvmsurveysim',
          install_requires=install_requires,
          include_package_data=True,
          packages=find_packages(),
          # package_dir={'': './'},
          scripts=[],
          classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Natural Language :: English',
              'Operating System :: OS Independent',
              'Programming Language :: Python',
              'Programming Language :: Python :: 3.6',
              'Topic :: Documentation :: Sphinx',
              'Topic :: Scientific/Engineering :: Astronomy',
              'Topic :: Software Development :: Libraries :: Python Modules',
              'Topic :: Software Development :: User Interfaces'
          ],
          )


if __name__ == '__main__':

    # Runs distutils
    run()
