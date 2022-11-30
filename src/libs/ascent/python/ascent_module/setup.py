###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# file: setup.py
# Purpose: disutils setup for ascent python module.
#
###############################################################################

import sys
from distutils.core import setup
from distutils.command.install_egg_info import install_egg_info

# disable install_egg_info
class SkipEggInfo(install_egg_info):
    def run(self):
        pass


setup (name = 'ascent',
       description = 'ascent',
       package_dir = {'ascent':'py_src'},
       packages=['ascent', 'ascent.bridge_kernel' , 'ascent.mpi'],
       cmdclass={'install_egg_info': SkipEggInfo})


