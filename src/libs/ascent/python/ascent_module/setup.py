###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# file: setup.py
# Purpose: setuptools setup for ascent python module.
#
###############################################################################

from setuptools import setup

setup (name = 'ascent',
       description = 'ascent',
       package_dir = {'ascent':'py_src'},
       zip_safe=False,
       packages=['ascent', 'ascent.bridge_kernel' , 'ascent.mpi'])


