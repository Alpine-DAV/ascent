###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# file: setup.py
# Purpose: setuptools setup for flow python module.
#
###############################################################################

from setuptools import setup
import sys

setup (name = 'flow',
       description = 'flow',
       package_dir = {'flow':'py_src'},
       zip_safe=False,
       packages=['flow'])



