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
import platform

cmake_rpath_settings = """
################################
# RPath Settings
################################

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

"""

class Icet(Package):
    homepage = "http://icet.sandia.gov/"
    url      = "https://gitlab.kitware.com/icet/icet"

    version('icet-master', git='https://gitlab.kitware.com/icet/icet.git')

    variant("mpich",default=False,description="build mpich as MPI lib for Icet")

    depends_on("cmake")

    if "darwin" in platform.system().lower():
        depends_on("mpich")

    depends_on("mpich",when="+mpich")

    def install(self, spec, prefix):
        # patch problem with cpack setup in the main CMakeLists.txt file
        # it looks like we can't use spack's filter_file, b/c the verbatium 
        # cmake syntax I am changing conflicts with python's regex syntax
        patched = open("CMakeLists.txt").read().replace('${ICET_SOURCE_DIR}/README)',
                                                        '${ICET_SOURCE_DIR}/README.md)')
        
        # I tried to get fPIC to work for static libs, but this didn't work
        #patched = patched.replace("PROJECT(ICET C)\n",
        #                          "PROJECT(ICET C)\n" + \nset(CMAKE_POSITION_INDEPENDENT_CODE ON)\n\n")

        #
        # hard code some RPATH settings:
        # spack settings weren't working with intel compilers
        # 
        patched = patched.replace("PROJECT(ICET C)\n",
                                  "PROJECT(ICET C)\n"+cmake_rpath_settings)
       
        open("CMakeLists.txt","w").write(patched)
        
        # build and install ice-t
        with working_dir('spack-build', create=True):
            mpicc  = which("mpicc")
            mpicxx = which("mpicxx")
            if mpicc is None or mpicxx is None:
                print "icet needs mpi ..."
                crash()
            cmake_args = ["..",
                          "-DBUILD_SHARED_LIBS=ON",
                          "-DICET_USE_OPENGL=OFF",
                          "-DMPI_C_COMPILER=%s" % mpicc.command,
                          "-DMPI_CXX_COMPILER=%s" % mpicxx.command]
            cmake_args.extend(std_cmake_args)
            print cmake_args
            cmake(*cmake_args)
            make()
            make("install")
            #make("test")
            



  
