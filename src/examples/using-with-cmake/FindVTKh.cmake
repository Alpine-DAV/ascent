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
# Setup VTKm 
#
###############################################################################
#
#  Expects VTKM_DIR to point to a Conduit installations.
#
# This file defines the following CMake variables:
#  VTKM_FOUND - If Conduit was found
#  VTKM_INCLUDE_DIRS - The Conduit include directories
#
#  If found, the vtkm CMake targets will also be imported.
#  The main vtkm library targets are:
#   vtkm 
#   vtkm_cont
#   vtkm_rendering
#
###############################################################################

###############################################################################
# Check for VTKM_DIR
###############################################################################
if(NOT VTKH_DIR)
  MESSAGE(FATAL_ERROR "Could not find VTKH_DIR. Conduit requires explicit VTKH_DIR.")
endif()

if(NOT EXISTS ${VTKH_DIR}/lib/VTKhConfig.cmake)
  MESSAGE(FATAL_ERROR "Could not find VTKh CMake include file (${VTKH_DIR}/lib/VTKhConfig.cmake)")
endif()

###############################################################################
# Import VTKm CMake targets
###############################################################################
include(${VTKH_DIR}/lib/VTKhConfig.cmake)

###############################################################################
# Set remaning CMake variables 
###############################################################################
# we found VTKh
set(VTKH_FOUND TRUE)
# provide location of the headers in VTKM_INCLUDE_DIRS
set(VTKH_INCLUDE_DIRS ${VTKH_DIR}/include/)




