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
# Setup Conduit
# This file defines:
#  CONDUIT_FOUND - If Conduit was found
#  CONDUIT_INCLUDE_DIRS - The Conduit include directories
#
#  If found, the conduit CMake targets will also be imported
###############################################################################

include(CMakeFindDependencyMacro)

# first Check for CONDUIT_DIR

if(NOT CONDUIT_DIR)
    MESSAGE(FATAL_ERROR "Conduit support needs explicit CONDUIT_DIR")
endif()

MESSAGE(STATUS "Looking for Conduit using CONDUIT_DIR = ${CONDUIT_DIR}")

###############################################################################
# Import Conduit's CMake targets
###############################################################################
find_dependency(Conduit REQUIRED
                NO_DEFAULT_PATH
                PATHS ${CONDUIT_DIR}/lib/cmake)

# if fortran is enabled, make sure conduit was built with fortran support
if(FORTRAN_FOUND)
    if(EXISTS ${CONDUIT_DIR}/include/conduit/conduit.mod)
        message(STATUS "FOUND conduit fortran module at: ${CONDUIT_DIR}/include/conduit/conduit.mod")
    else()
        message(FATAL_ERROR "Failed to find conduit fortran module. \
                             Ascent Fortran support requires Conduit with Fortran enabled.")
    endif()
endif()


if(EXISTS ${CONDUIT_DIR}/include/conduit/conduit_relay_hdf5_api.hpp)
    set(CONDUIT_HDF5_ENABLED TRUE)
    message(STATUS "FOUND conduit HDF5 support")
endif()


if(EXISTS ${CONDUIT_DIR}/include/conduit/conduit_relay_adios_api.hpp)
    set(CONDUIT_ADIOS_ENABLED TRUE)
    message(STATUS "FOUND conduit ADIOS support")
endif()

set(CONDUIT_FOUND TRUE)
set(CONDUIT_INCLUDE_DIRS ${CONDUIT_DIR}/include/conduit)

if(PYTHON_FOUND)
    find_package(PythonInterp)
    if(PYTHONINTERP_FOUND)
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
                                "import os; import conduit; print(os.path.split(conduit.__file__)[0]);"
                        RESULT_VARIABLE _FIND_CONDUIT_PYTHON_RESULT
                        OUTPUT_VARIABLE _FIND_CONDUIT_PYTHON_OUT
                        ERROR_VARIABLE  _FIND_CONDUIT_PYTHON_ERROR_VALUE
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
                      message(STATUS "PYTHOn found!! ${CONDUIT_DIR}")
        if(_FIND_CONDUIT_PYTHON_RESULT MATCHES 0)
            message(STATUS "FOUND conduit python module at: ${_FIND_CONDUIT_PYTHON_OUT}")
        else()
            # try find the path to the conduit python module assuming a standard conduit install
            execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
                                   "import sys; import os; sys.path.append(os.path.join('${CONDUIT_DIR}','python-modules'));  import conduit; print(os.path.split(conduit.__file__)[0]);"
                                    RESULT_VARIABLE _FIND_CONDUIT_PYTHON_RESULT
                                    OUTPUT_VARIABLE _FIND_CONDUIT_PYTHON_OUT
                                    ERROR_VARIABLE  _FIND_CONDUIT_PYTHON_ERROR_VALUE
                                    OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(_FIND_CONDUIT_PYTHON_RESULT MATCHES 0)
                # we will use this to make sure we can setup tests correctly
                set(EXTRA_PYTHON_MODULE_DIRS "${CONDUIT_DIR}/python-modules/")
                message(STATUS "FOUND conduit python module at: ${_FIND_CONDUIT_PYTHON_OUT}")
            else()
                message(FATAL_ERROR
                "conduit python import failure:\n${_FIND_CONDUIT_PYTHON_OUT}")

            endif()
        endif()
    else()
        message(FATAL_ERROR "PYTHON_FOUND = TRUE, but could not find a python interpreter.")
    endif()

    set(CONDUIT_PYTHON_INCLUDE_DIR ${_FIND_CONDUIT_PYTHON_OUT})
    message(STATUS "FOUND conduit python include dir: ${CONDUIT_PYTHON_INCLUDE_DIR}")
    list(APPEND CONDUIT_INCLUDE_DIRS ${CONDUIT_PYTHON_INCLUDE_DIR})
endif()

message(STATUS "FOUND Conduit at ${CONDUIT_DIR}")
message(STATUS "CONDUIT_INCLUDE_DIRS = ${CONDUIT_INCLUDE_DIRS}")


blt_register_library( NAME conduit
                      INCLUDES ${CONDUIT_INCLUDE_DIRS}
                      LIBRARIES  conduit conduit_relay conduit_blueprint)

# assumes conduit was built with mpi
if(MPI_FOUND)
    blt_register_library( NAME conduit_relay_mpi
                          INCLUDES ${CONDUIT_INCLUDE_DIRS}
                          LIBRARIES  conduit_relay_mpi)
endif()

