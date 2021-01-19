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
# Setup ADIOS
#
###############################################################################

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()


# first Check for ADIOS_DIR

if(NOT ADIOS2_DIR)
    MESSAGE(FATAL_ERROR "ADIOS2 support needs explicit ADIOS2_DIR")
endif()

MESSAGE(STATUS "*************************************** MEOW **************************************")
MESSAGE(STATUS "*************************************** MEOW **************************************")
MESSAGE(STATUS "*************************************** MEOW **************************************")
MESSAGE(STATUS "Looking for ADIOS2 using ADIOS2_DIR = ${ADIOS2_DIR}")

find_package(ADIOS2 REQUIRED)

#set(ADIOS2_INCLUDE_DIR ${ADIOS2_DIR}/include)
#set(ADIOS2_LIB_DIR ${ADIOS2_DIR}/lib)

#set(ADIOS2_INCLUDE_DIRS /apps/ADIOS/install_par/include)
#set(ADIOS2_LIB_DIRS /apps/ADIOS/install_par/lib)

message(status "*********** AIDOS 2 stuff ${ADIOS2_DIR}")
message(status "*********** AIDOS 2 stuff ${ADIOS2_INCLUDE_DIRS}")
message(status "*********** AIDOS 2 stuff ${ADIOS2_LIB_DIRS}")
message(status "*********** AIDOS 2 stuff ${ADIOS2_LIBRARIES}")
message(status "************* ALLL VARS ************************")
#print_all_variables()

#blt_register_library(NAME adios2
#                     INCLUDES ${ADIOS2_INCLUDE_DIR}
#                     LIBRARIES ${ADIOS2_LIB_DIRS} ${ADIOS2_LIBRARIES} )













# # CMake's FindADIOS2 module uses the ADIOS2_ROOT env var
# set(ADIOS2_ROOT ${ADIOS2_DIR})
# set(ENV{ADIOS2_ROOT} ${ADIOS2_ROOT})

# # Use CMake's FindADIOS2 module, which uses hdf5's compiler wrappers to extract
# # all the info about the hdf5 install
# include(${ADIOS2_DIR}/etc/FindADIOS2.cmake)

# # FindADIOS2 sets ADIOS2_DIR to it's installed CMake info if it exists
# # we want to keep ADIOS2_DIR as the root dir of the install to be
# # consistent with other packages

# set(ADIOS2_DIR ${ADIOS2_ROOT} CACHE PATH "" FORCE)
# # not sure why we need to set this, but we do
# #set(ADIOS2_FOUND TRUE CACHE PATH "" FORCE)

# if(NOT ADIOS2_FOUND)
#     message(FATAL_ERROR "ADIOS2_DIR is not a path to a valid ADIOS2 install")
# endif()

# blt_register_library(NAME adios
#                      INCLUDES ${ADIOS2_INCLUDE_DIRS}
#                      LIBRARIES ${ADIOS2_LIBRARIES} )
