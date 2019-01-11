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
# Setup Ascent
#
###############################################################################
#
#  Expects ASCENT_DIR to point to a Ascent installation.
#
# This file defines the following CMake variables:
#  ASCENT_FOUND - If Ascent was found
#  ASCENT_INCLUDE_DIRS - The Conduit include directories
#
#  If found, the ascent CMake targets will also be imported.
#  The main ascent library targets are:
#   ascent
#   ascent_mpi (if ascent was built with mpi support)
#
###############################################################################

###############################################################################
# Check for ASCENT_DIR
###############################################################################
if(NOT ASCENT_DIR)
    MESSAGE(FATAL_ERROR "Could not find Ascent. Ascent requires explicit ASCENT_DIR.")
endif()

get_filename_component(ASCENT_DIR ${ASCENT_DIR} ABSOLUTE)

if(NOT EXISTS ${ASCENT_DIR}/lib/cmake/ascent.cmake)
    MESSAGE(FATAL_ERROR "Could not find Ascent CMake include file (${ASCENT_DIR}/lib/cmake/ascent.cmake)")
endif()

###############################################################################
# Import Ascent's CMake targets
###############################################################################
include(${ASCENT_DIR}/lib/cmake/ascent-config.cmake)

###############################################################################
# Set remaning CMake variables
###############################################################################
# we found Ascent
set(ASCENT_FOUND TRUE)

# If ZZZ_DIR not set, use known install path for Conduit, VTK-h and VTK-m
# This will be picked up in by FindZZZ
if(NOT CONDUIT_DIR)
    set(CONDUIT_DIR ${ASCENT_CONDUIT_DIR})
endif()

if(NOT VTKH_DIR)
    set(VTKH_DIR ${ASCENT_VTKH_DIR})
endif()

if(NOT VTKM_DIR)
    set(VTKM_DIR ${ASCENT_VTKM_DIR})
endif()




