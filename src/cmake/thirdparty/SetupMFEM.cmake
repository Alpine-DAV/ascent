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
# Setup MFEM
#
###############################################################################

# first Check for MFEM_DIR

if(NOT MFEM_DIR)
    MESSAGE(FATAL_ERROR "MFEM support needs explicit MFEM_DIR")
endif()

MESSAGE(STATUS "Looking for MFEM using MFEM_DIR = ${MFEM_DIR}")


# when mfem is built w/o cmake, we can get the details of deps from its
# config.mk file
find_path(MFEM_CFG_DIR config.mk
          PATHS ${MFEM_DIR}/share/mfem/
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

if(NOT MFEM_CFG_DIR)
    MESSAGE(FATAL_ERROR "Failed to find MFEM share/mfem/config.mk")
endif()

# read config.mk file
file(READ "${MFEM_CFG_DIR}/config.mk" mfem_cfg_file_txt)

# parse include flags
string(REGEX MATCHALL "MFEM_TPLFLAGS .+\n" mfem_tpl_inc_flags ${mfem_cfg_file_txt})
string(REGEX REPLACE  "MFEM_TPLFLAGS +=" "" mfem_tpl_inc_flags ${mfem_tpl_inc_flags})
string(FIND  ${mfem_tpl_inc_flags} "\n" mfem_tpl_inc_flags_end_pos)
string(SUBSTRING ${mfem_tpl_inc_flags} 0 ${mfem_tpl_inc_flags_end_pos} mfem_tpl_inc_flags)
string(STRIP ${mfem_tpl_inc_flags} mfem_tpl_inc_flags)

# parse link flags
string(REGEX MATCHALL "MFEM_EXT_LIBS .+\n" mfem_tpl_lnk_flags ${mfem_cfg_file_txt})
string(REGEX REPLACE  "MFEM_EXT_LIBS +=" "" mfem_tpl_lnk_flags ${mfem_tpl_lnk_flags})
string(FIND  ${mfem_tpl_lnk_flags} "\n" mfem_tpl_lnl_flags_end_pos )
string(SUBSTRING ${mfem_tpl_lnk_flags} 0 ${mfem_tpl_lnl_flags_end_pos} mfem_tpl_lnk_flags)
string(STRIP ${mfem_tpl_lnk_flags} mfem_tpl_lnk_flags)

# make sure mfem was built with with conduit support:
message(STATUS "Checking for MFEM conduit support")
string(REGEX MATCHALL "MFEM_USE_CONDUIT += +YES" mfem_use_conduit ${mfem_cfg_file_txt})
if(mfem_use_conduit STREQUAL "")
    message(FATAL_ERROR "MFEM config.mk missing MFEM_USE_CONDUIT = YES")
else()
    message(STATUS "Found MFEM_USE_CONDUIT = YES in MFEM config.mk")
endif()

#find includes
find_path(MFEM_INCLUDE_DIRS mfem.hpp
          PATHS ${MFEM_DIR}/include
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

#find libs
find_library(MFEM_LIBRARIES LIBRARIES NAMES mfem
             PATHS ${MFEM_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)


if(MFEM_LIBRARIES)
    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set MFEM_FOUND to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(MFEM DEFAULT_MSG
                                      MFEM_LIBRARIES MFEM_INCLUDE_DIRS)
endif()


if(NOT MFEM_FOUND)
    message(FATAL_ERROR "MFEM_FOUND is not a path to a valid MFEM install")
endif()

    # assume mfem is built with mpi support for now
blt_register_library(NAME mfem
                     INCLUDES ${MFEM_INCLUDE_DIRS}
                     COMPILE_FLAGS ${mfem_tpl_inc_flags}
                     LIBRARIES ${MFEM_LIBRARIES} ${mfem_tpl_lnk_flags})
