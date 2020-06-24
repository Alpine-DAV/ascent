###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
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

###############################################################################
# Setup VTK
###############################################################################


######## FIX LATER ########

set(TP_HOME "/g/g92/shudler1/projects/visit/third_party")
set(ARCH "linux-x86_64_gcc-4.9")
set(VTK_DIR "${TP_HOME}/vtk/8.1.0/${ARCH}/lib/cmake/vtk-8.1")

link_directories("${TP_HOME}/llvm/5.0.0/${ARCH}/lib")
link_directories("${TP_HOME}/mesagl/17.2.8/${ARCH}/lib")

###########################


if(NOT VTK_DIR)
    MESSAGE(FATAL_ERROR "VTK support needs explicit VTK_DIR")
endif()

MESSAGE(STATUS "Looking for VTK using VTK_DIR = ${VTK_DIR}")

# use VTK_DIR to setup the options that cmake's find VTK needs
#file(GLOB VTK_DIR "${VTK_DIR}/lib/cmake/vtk-*")
#if(NOT VTK_DIR)
#    MESSAGE(FATAL_ERROR "Failed to find VTK at VTK_DIR=${VTK_DIR}/lib/cmake/vtk-*")
#endif()

find_package(VTK REQUIRED QUIET)
include(${VTK_USE_FILE})

set(VTK_FOUND TRUE)

message(STATUS "FOUND VTK at ${VTK_DIR}")

message(STATUS "VTK include dirs: ${VTK_INCLUDE_DIRS}")
message(STATUS "VTK library dirs: ${VTK_LIBRARIES}")

blt_register_library( NAME vtk
					  INCLUDES ${VTK_INCLUDE_DIRS}
                      LIBRARIES ${VTK_LIBRARIES}
                      )
