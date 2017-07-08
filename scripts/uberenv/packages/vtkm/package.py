###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Alpine. 
# 
# For details, see: http://software.llnl.gov/alpine/.
# 
# Please also read alpine/LICENSE
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

class Vtkm(Package):
    homepage = "https://m.vtk.org/"
    url      = "http://m.vtk.org/images/8/87/Vtk-m-1.0.0.tar.gz"

    version('kitware-gitlab',
            git='https://gitlab.kitware.com/vtk/vtk-m.git',
            branch='master')

    #version('kitware-gitlab-test',
    #        git='https://gitlab.kitware.com/mclarsen/vtk-m.git',
    #        branch='cyrush_test_me')

    #version('1.0.0',  '9d9d45e675d5b0628b19b32f5542ed9c')

    depends_on("cmake")
    depends_on("tbb")
    #depends_on("boost-headers")
    #patch('vtkm_patch.patch')
    def install(self, spec, prefix):
        os.environ["TBB_ROOT"] = spec["tbb"].prefix
        with working_dir('spack-build', create=True):
            cmake_args = ["../",
                          "-DVTKm_ENABLE_TBB=ON",
                          "-DVTKm_ENABLE_TESTING=OFF",
                          "-DVTKm_BUILD_RENDERING=ON",
                          "-DVTKm_USE_64BIT_IDS=OFF",
                          "-DVTKm_USE_DOUBLE_PRECISION=ON"]
            # check for cuda support
            nvcc = which("nvcc")
            if not nvcc  is None:
                cmake_args.append("-DVTKm_ENABLE_CUDA=ON")
                # this fix is necessary if compiling platform has cuda, but no devices
                # (this common for front end nodes on hpc clusters)
                # we choose kepler for llnl surface and ornl titan
                cmake_args.append("-DVTKm_CUDA_Architecture=kepler")
            cmake_args.extend(std_cmake_args)
            print cmake_args
            cmake(*cmake_args)
            make(parallel=False)
            make("install",parallel=False)



