# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Babelflow(CMakePackage):
    """FIXME: Put a proper description of your package here."""

    homepage = "https://github.com/sci-visus/BabelFlow"
    url      = "https://github.com/sci-visus/BabelFlow/archive/v1.0.0.tar.gz"

    maintainers = ['spetruzza']

    version('1.0.0',  sha256='4c4d7ddf60e25e8d3550c07875dba3e46e7c9e61b309cc47a409461b7ffa405e')

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
