# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack import *

#import sys
#import os
#import socket
#import llnl.util.tty as tty
#from os import environ as env

class Fides(CMakePackage) :
    """A library that provides a schema for ADIOS2 streams.
    """
    homepage = "https://gitlab.kitware.com/vtk/fides"
    url      = "https://gitlab.kitware.com/vtk/fides/-/archive/v1.0.0/fides-v1.0.0.tar.gz"
    git      = "https://gitlab.kitware.com/vtk/fides.git"

    maintainers = ['caitlin.ross', 'dpugmire']

    version('master', branch='master', default=True)
#    version('1.0.0', sha256='c355fdb4ca3790c1fa9a4491a0d294b8f883b6946c540ad9e5633c9fd8c8c3aa')

    variant("mpi", default=True, description="build mpi support")
    variant("adios2", default=True)  ##this adds +adios2 implicitly
    variant('vtk-m', default=True)

    # Certain CMake versions have been found to break for our use cases
    depends_on("cmake@3.14.1:3.14.99,3.18.2:", type='build')

    depends_on('adios2~zfp', when='+adios2')
    depends_on("mpi", when="+mpi")
    depends_on("vtk-m", when="+vtk-m")

    def cmake_args(self):
        spec = self.spec
        options = []

        options.append("-DVTKm_DIR={0}".format(spec['vtk-m'].prefix))
        options.append("-DADIOS2_DIR={0}".format(spec['adios2'].prefix))
        options.append("-DFIDES_ENABLE_TESTING=OFF")
        options.append("-DFIDES_ENABLE_EXAMPLES=OFF")

        print('FIDES*********************************************************')
        print('FIDES*********************************************************')
        print('FIDES*********************************************************')
        print( 'FIDES options=', options)

        return options
