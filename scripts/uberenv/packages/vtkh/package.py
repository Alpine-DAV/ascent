###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Ascent. 
# 
# For details, see: http://software.llnl.gov/ascent/.
# 
# Please also read ascent/LICENSE
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
# 
###############################################################################
from spack import *
import os
import platform

class Vtkh(Package):
    homepage = "https://h.vtk.org/"
    url      = "https://github.com/Alpine-DAV/vtk-h"

    version('vtkh-reorg',
            git='https://github.com/Alpine-DAV/vtk-h.git',
            branch='master',
            submodules=True)

    variant("mpich",default=False,description="build with mpich")

    depends_on("cmake")
    depends_on("tbb")
    depends_on("vtkm")

    if "darwin" in platform.system().lower():
        depends_on("mpich")

    depends_on("mpich",when="+mpich")

    def install(self, spec, prefix):
        # spack version used doesn't support submodules, so manually update it
        # here.
        os.system("git submodule update --init")

        with working_dir('spack-build', create=True):
            cmake_args = ["../src",
                          "-DVTKM_DIR=%s" % spec["vtkm"].prefix,
                          "-DTBB_DIR=%s"  % spec["tbb"].prefix,
                          "-DENABLE_TESTS=OFF",
                          "-DBUILD_TESTING=OFF"]
            if "+mpich" in spec:
                mpicc  = which("mpicc")
                mpicxx = which("mpicxx")
                if mpicc is None or mpicxx is None:
                    print "VTKh needs mpi ..."
                    crash()
                cmake_args.extend([
                          "-DMPI_C_COMPILER=%s" % mpicc.command,
                          "-DMPI_CXX_COMPILER=%s" % mpicxx.command])

            # check for cuda support
            nvcc = which("nvcc")
            if not nvcc  is None:
                cmake_args.append("-DENABLE_CUDA=ON")
                # this fix is necessary if compiling platform has cuda, but no devices
                # (this common for front end nodes on hpc clusters)
                # we choose kepler for llnl surface and ornl titan
                cmake_args.append("-DVTKm_CUDA_Architecture=kepler")
            cmake_args.extend(std_cmake_args)
            print cmake_args
            cmake(*cmake_args)
            make()
            make("install",parallel=False)
