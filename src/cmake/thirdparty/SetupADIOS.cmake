###############################################################################
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
# Setup ADIOS
#
###############################################################################

# first Check for ADIOS_DIR

if(NOT ADIOS_DIR)
    MESSAGE(FATAL_ERROR "ADIOS support needs explicit ADIOS_DIR")
endif()

MESSAGE(STATUS "Looking for ADIOS using ADIOS_DIR = ${ADIOS_DIR}")

# CMake's FindADIOS module uses the ADIOS_ROOT env var
set(ADIOS_ROOT ${ADIOS_DIR})
set(ENV{ADIOS_ROOT} ${ADIOS_ROOT})

# Use CMake's FindADIOS module, which uses hdf5's compiler wrappers to extract
# all the info about the hdf5 install
include(${ADIOS_DIR}/etc/FindADIOS.cmake)

# FindADIOS sets ADIOS_DIR to it's installed CMake info if it exists
# we want to keep ADIOS_DIR as the root dir of the install to be 
# consistent with other packages

set(ADIOS_DIR ${ADIOS_ROOT} CACHE PATH "" FORCE)
# not sure why we need to set this, but we do
#set(ADIOS_FOUND TRUE CACHE PATH "" FORCE)

if(NOT ADIOS_FOUND)
    message(FATAL_ERROR "ADIOS_DIR is not a path to a valid ADIOS install")
endif()

blt_register_library(NAME adios
                     INCLUDES ${ADIOS_INCLUDE_DIRS}
                     LIBRARIES ${ADIOS_LIBRARIES} )

