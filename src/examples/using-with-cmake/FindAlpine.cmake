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

###############################################################################
#
# Setup Alpine
#
###############################################################################
#
#  Expects ALPINE_DIR to point to a Alpine installation.
#
# This file defines the following CMake variables:
#  ALPINE_FOUND - If Alpine was found
#  ALPINE_INCLUDE_DIRS - The Conduit include directories
#
#  If found, the alpine CMake targets will also be imported.
#  The main alpine library targets are:
#   alpine
#   alpine_par (if alpine was built with mpi support)
#
###############################################################################

###############################################################################
# Check for ALPINE_DIR
###############################################################################
if(NOT ALPINE_DIR)
    MESSAGE(FATAL_ERROR "Could not find Alpine. Alpine requires explicit ALPINE_DIR.")
endif()

if(NOT EXISTS ${ALPINE_DIR}/lib/cmake/alpine.cmake)
    MESSAGE(FATAL_ERROR "Could not find Alpine CMake include file (${ALPINE_DIR}/lib/cmake/alpine.cmake)")
endif()

###############################################################################
# Import Alpine's CMake targets
###############################################################################
include(${ALPINE_DIR}/lib/cmake/alpine.cmake)

###############################################################################
# Set remaning CMake variables 
###############################################################################
# we found Alpine
set(ALPINE_FOUND TRUE)
# provide location of the headers in ALPINE_INCLUDE_DIRS
set(ALPINE_INCLUDE_DIRS ${ALPINE_DIR}/include/alpine)




