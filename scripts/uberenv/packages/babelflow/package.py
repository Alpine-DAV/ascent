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


class Babelflow(CMakePackage):
    """FIXME: Put a proper description of your package here."""

    homepage = "https://github.com/sci-visus/BabelFlow"
    url      = "https://github.com/sci-visus/BabelFlow/archive/ascent.zip"

    maintainers = ['spetruzza']

    version('develop',
            git='https://github.com/sci-visus/BabelFlow.git',
            branch='ascent',
            submodules=True,
            preferred=True)

    depends_on('mpi')

    variant("shared", default=True, description="Build Babelflow as shared libs")

    def cmake_args(self):
      args = []

      #args.append('-DMPI_C_COMPILER='+self.spec['mpi'].mpicc)
      #args.append('-DMPI_CXX_COMPILER='+self.spec['mpi'].mpicxx)

      return args
  
    def cmake_install(self, spec, prefix):
        #print(cmake_cache_entry("MPI_C_COMPILER",spec['mpi'].mpicc))
        
        if "+shared" in spec:
            cmake_args.append('-DBUILD_SHARED_LIBS=ON')
        else:
            cmake_args.append('-DBUILD_SHARED_LIBS=OFF')
            
        make()
        make('install')
