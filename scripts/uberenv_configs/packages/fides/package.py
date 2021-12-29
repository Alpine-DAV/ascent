# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Fides(CMakePackage):
    """A library that provides a schema for ADIOS2 streams."""
    homepage = "https://gitlab.kitware.com/vtk/fides"
    url      = "https://gitlab.kitware.com/vtk/fides/-/archive/v1.0.0/fides-v1.0.0.tar.gz"
    git      = "https://gitlab.kitware.com/vtk/fides.git"

    maintainers = ['caitlinross', 'dpugmire']

    version('master', branch='master')
    version('1.1.0', sha256='40d2e08b8d5cfdfc809eae6ed2ae0731108ce3b1383485f4934a5ec8aaa9425e')
    version('1.0.0', sha256='c355fdb4ca3790c1fa9a4491a0d294b8f883b6946c540ad9e5633c9fd8c8c3aa')

    variant("mpi", default=True, description="build mpi support")
    variant("adios2", default=True, description="build ADIOS2 support")
    variant('vtk-m', default=True, description="build VTK-m support")

    # Certain CMake versions have been found to break for our use cases
    depends_on("cmake@3.14.1:3.14.99,3.18.2:", type='build')

    depends_on("mpi", when="+mpi")
    depends_on('adios2~zfp', when='+adios2')
    depends_on("vtk-m", when="+vtk-m")

    # Fix missing implict includes
    @when('%gcc@10:')
    def setup_build_environment(self, env):
        env.append_flags('CXXFLAGS', '-include limits -include numeric')


    def cmake_args(self):
        spec = self.spec
        options = [
            self.define("VTKm_DIR", spec['vtk-m'].prefix),
            self.define("ADIOS2_DIR", spec['adios2'].prefix),
            self.define("FIDES_ENABLE_TESTING", "OFF"),
            self.define("FIDES_ENABLE_EXAMPLES", "OFF")
        ]
        return options
