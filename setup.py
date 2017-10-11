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

from setuptools import setup

import os
import warnings


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def convert_md_to_rst(fp):
    try:
        import pypandoc
        output = pypandoc.convert_file(fp, 'rst')
        return output
    except ImportError:
        warnings.warn('cannot import pypandoc.', UserWarning)
        return open(fp).read()


requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_requires = [line.strip().replace('==', '>=') for line in open(requirements_file)
                    if not line.strip().startswith('#') and line.strip() != '']

NAME = 'sdss-pydrp'
# do not use x.x.x-dev.  things complain.  instead use x.x.xdev
VERSION = '0.1.0dev'
RELEASE = 'dev' not in VERSION


def run(data_files, packages):

    setup(name=NAME,
          version=VERSION,
          license='BSD3',
          description='A data reduction pipeline for MaNGA data, in Python.',
          long_description=convert_md_to_rst('README.md'),
          author='José Sánchez-Gallego',
          author_email='gallegoj@uw.edu',
          keywords='MaNGA DRP pipeline',
          url='https://github.com/albireox/pydrp',
          install_requires=install_requires,
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
