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
# Setup IceT
# This file defines:
#  ICET_FOUND - If IceT was found
#  ICET_INCLUDE_DIRS - The IceT include directories
#  ICET_LIBRARIES - The libraries needed to use IceT
###############################################################################

# first Check for ICET_DIR

if(NOT ICET_DIR)
    MESSAGE(FATAL_ERROR "IceT support needs explicit ICET_DIR")
endif()

MESSAGE(STATUS "Looking for IceT using ICET_DIR = ${ICET_DIR}")

#find includes
find_path(ICET_INCLUDE_DIRS IceT.h
          PATHS ${ICET_DIR}/include
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

#find libs
find_library(ICET_CORE_LIB LIBRARIES NAMES IceTCore
             PATHS ${ICET_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)


find_library(ICET_MPI_LIB LIBRARIES NAMES IceTMPI
             PATHS ${ICET_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)


set(ICET_LIBRARIES ${ICET_CORE_LIB} ${ICET_MPI_LIB})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ICET_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(IceT  DEFAULT_MSG
                                  ICET_LIBRARIES ICET_INCLUDE_DIRS)

mark_as_advanced(ICET_CORE_LIB
                 ICET_MPI_LIB)

if(NOT ICET_FOUND)
    message(FATAL_ERROR "ICET_DIR is not a path to a valid icet install")
endif()

blt_register_library(NAME icet
                     INCLUDES ${ICET_INCLUDE_DIRS}
                     LIBRARIES ${ICET_LIBRARIES} )

