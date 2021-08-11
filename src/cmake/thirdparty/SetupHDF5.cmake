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
#
# Setup HDF5
#
###############################################################################

# first Check for HDF5_DIR

if(NOT HDF5_DIR)
    MESSAGE(FATAL_ERROR "HDF5 support needs explicit HDF5_DIR")
endif()

MESSAGE(STATUS "Looking for HDF5 using HDF5_DIR = ${HDF5_DIR}")

# CMake's FindHDF5 module uses the HDF5_ROOT env var
set(HDF5_ROOT ${HDF5_DIR})

if(NOT WIN32)
    # use HDF5_ROOT env var for FindHDF5 with older versions of cmake
    if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
        set(ENV{HDF5_ROOT} ${HDF5_ROOT}/bin)
    endif()
endif()

# Use CMake's FindHDF5 module, which uses hdf5's compiler wrappers to extract
# all the info about the hdf5 install
include(FindHDF5)

# FindHDF5 sets HDF5_DIR to it's installed CMake info if it exists
# we want to keep HDF5_DIR as the root dir of the install to be
# consistent with other packages

set(HDF5_DIR ${HDF5_ROOT} CACHE PATH "" FORCE)
# not sure why we need to set this, but we do
#set(HDF5_FOUND TRUE CACHE PATH "" FORCE)

if(NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5_DIR is not a path to a valid hdf5 install")
endif()

blt_register_library(NAME hdf5
                     INCLUDES ${HDF5_INCLUDE_DIRS}
                     LIBRARIES ${HDF5_LIBRARIES} )

