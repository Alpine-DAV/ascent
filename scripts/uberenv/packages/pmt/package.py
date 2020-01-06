# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install babelflow
#
# You can edit this file again by typing:
#
#     spack edit babelflow
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack import *


class Pmt(CMakePackage):
    """FIXME: Put a proper description of your package here."""

    homepage = "https://bitbucket.org/cedmav/parallelmergetree"
    url      = "https://bitbucket.org/cedmav/parallelmergetree/get/ascent.zip"

    maintainers = ['spetruzza']

    version('0.0.0', sha256='8f984f643a15107716cef0d146846359672823da9dd95aea4b5cccea0984d743')

    depends_on('uberenv-babelflow')

    def cmake_args(self):
      args = []

      args.append('-DLIBRARY_ONLY=ON')

      return args

    def cmake_install(self, spec, prefix):
        # FIXME: Unknown build system
        make()
        make('install')
