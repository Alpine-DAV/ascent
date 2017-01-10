###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Strawman. 
# 
# For details, see: http://software.llnl.gov/strawman/.
# 
# Please also read strawman/LICENSE
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
# Setup OSMesa
# This file defines:
#  OSMESA_FOUND - If OSMesa was found
#  OSMESA_INCLUDE_DIRS - The OSMesa include directories
#  OSMESA_LIBRARIES - The libraries needed to use OSMesa
###############################################################################

# first Check for OSMESA_DIR

if(NOT OSMESA_DIR)
    MESSAGE(FATAL_ERROR "OSMesa support needs explicit OSMESA_DIR")
endif()

MESSAGE(STATUS "Looking for OSMesa using OSMESA_DIR = ${OSMESA_DIR}")

#find includes
find_path(OSMESA_INCLUDE_DIRS GL
          PATHS ${OSMESA_DIR}/include
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

#find libs
find_library(OSMESA_LIBRARIES LIBRARIES NAMES OSMesa
             PATHS ${OSMESA_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OSMESA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OSMesa  DEFAULT_MSG
                                  OSMESA_LIBRARIES OSMESA_INCLUDE_DIRS)

if(NOT OSMESA_FOUND)
    message(FATAL_ERROR "OSMESA_DIR is not a path to a valid osmesa install")
endif()

