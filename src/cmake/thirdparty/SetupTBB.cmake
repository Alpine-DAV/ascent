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
# Setup TBB
###############################################################################

# Check for TBB_DIR
if(NOT TBB_DIR)
    MESSAGE(FATAL_ERROR "TBB support needs explicit TBB_DIR")
endif()

MESSAGE(STATUS "Looking for TBB using TBB_DIR = ${TBB_DIR}")

#find includes
find_path(TBB_INCLUDE_DIRS tbb/tbb.h
          PATHS ${TBB_DIR}/include
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

#find libs
find_library(TBB_LIB LIBRARIES NAMES tbb
             PATHS ${TBB_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)
#
find_library(TBB_MALLOC_LIB LIBRARIES NAMES tbbmalloc
             PATHS ${TBB_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

#
find_library(TBB_MALLOC_PROXY_LIB LIBRARIES NAMES tbbmalloc_proxy
             PATHS ${TBB_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

# setup libs var
set(TBB_LIBRARIES ${TBB_LIB})
# not sure if we need to use tbbmalloc or its proxy
# list(APPEND TBB_LIBRARIES ${TBB_MALLOC_LIB} ${TBB_MALLOC_PROXY_LIB})


#find debug libs
find_library(TBB_DEBUG_LIB LIBRARIES NAMES tbb_debug
             PATHS ${TBB_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)
#
find_library(TBB_MALLOC_DEBUG_LIB LIBRARIES NAMES tbbmalloc_debug
             PATHS ${TBB_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

#
find_library(TBB_MALLOC_PROXY_DEBUG_LIB LIBRARIES NAMES tbbmalloc_proxy_debug
             PATHS ${TBB_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

# setup debug libs var
set(TBB_DEBUG_LIBRARIES ${TBB_DEBUG_LIB})
# not sure if we need to use tbbmalloc or its proxy
#list(APPEND TBB_DEBUG_LIBRARIES ${TBB_MALLOC_DEBUG_LIB} ${TBB_MALLOC_PROXY_DEBUG_LIB})

if(TBB_LIBRARIES)
    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set TBB_FOUND to TRUE
    # if all listed variables are TRUE
        find_package_handle_standard_args(TBB DEFAULT_MSG
                                          TBB_LIBRARIES TBB_DEBUG_LIBRARIES TBB_INCLUDE_DIRS)
endif()

#
# If the above detection logic didn't work, try again with FindTBB.
#
if(NOT TBB_FOUND)
    set(TBB_INCLUDE_DIRS "")
    set(TBB_LIB "")
    set(TBB_MALLOC_LIB "")
    set(TBB_MALLOC_PROXY_LIB "")

    set(TBB_DEBUG_LIB "")
    set(TBB_MALLOC_DEBUG_LIB "")
    set(TBB_MALLOC_PROXY_DEBUG_LIB "")

    set(TBBROOT ${TBB_DIR})
    include(CMake/thirdparty/FindTBB.cmake)
endif()


if(NOT TBB_FOUND)
    message(FATAL_ERROR "TBB_DIR is not a path to a valid tbb install")
endif()

blt_register_library(NAME tbb
                     INCLUDES ${TBB_INCLUDE_DIRS}
                     LIBRARIES ${TBB_LIB} ${TBB_MALLOC_LIB} ${TBB_MALLOC_PROXY_LIB})

