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

################################
# cmake policy selection
################################
# avoid warnings for project commands w/o VERSION
cmake_policy(SET CMP0048 NEW)

# allow find_packages to use ZZZ_ROOT vars
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

################################
# Ascent
################################

project(ascent VERSION "0.7.1")

################################
# Build Options
################################
option(BUILD_SHARED_LIBS  "Build shared libraries"    ON)
option(ENABLE_TESTS       "Build tests"               ON)

option(ENABLE_FORTRAN     "Build Fortran support"     ON)
option(ENABLE_PYTHON      "Build Python Support"      ON)

option(ENABLE_MPI         "Build MPI Support"         ON)
option(ENABLE_SERIAL      "Build Serial (non-MPI) Support" ON)
option(ENABLE_CUDA        "Build CUDA Support"        OFF)
option(ENABLE_OPENMP      "Build OpenMP Support"      OFF)

option(ENABLE_EXAMPLES    "Build Examples"            ON)
option(ENABLE_UTILS       "Build Utilities"           ON)
option(ENABLE_DOCS        "Build Documentation"       ON)
option(ENABLE_MFEM        "Enable MFEM support"       ON)
option(ENABLE_FIDES       "Build with Fides"          OFF)

option(ENABLE_BABELFLOW   "Enable BabelFlow runtime"  OFF)

if(NOT ENABLE_SERIAL AND NOT ENABLE_MPI)
  message(FATAL_ERROR "No libraries are built. "
    "Please set ENABLE_SERIAL, ENABLE_MPI or both to ON")
endif()


if(ENABLE_CUDA)
  cmake_minimum_required(VERSION 3.14)
else()
  cmake_minimum_required(VERSION 3.9)
endif()

if(ENABLE_CUDA AND BUILD_SHARED_LIBS)
  message(FATAL_ERROR "Static libraries are required when building with CUDA")
endif()

if(ENABLE_CUDA)
    # ask politely a few times for SEPARABLE COMPILATION
    set (CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )
    set (CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )
endif()

################################
# cmake policy selection
################################
# cmake 3.18+ use CMP0104 OLD to manually specify cuda flags
#https://cmake.org/cmake/help/latest/policy/CMP0104.html
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()


################################
# Populate the cmake gui
################################
set(CONDUIT_DIR "" CACHE PATH "path to conduit installation")

################################
# Invoke CMake Fortran setup
# if ENABLE_FORTRAN == ON
################################
if(ENABLE_FORTRAN)
    enable_language(Fortran)
endif()


################################
# Set BLT Options
################################
# don't use BLT's all warnings feature
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "")

if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++11" CACHE STRING "")
endif()

# Init BLT
################################
# if BLT_SOURCE_DIR is not set - use "blt" as default
if(NOT BLT_SOURCE_DIR)
    set(BLT_SOURCE_DIR "blt")
endif()

# init blt using BLT_SOURCE_DIR
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

################################
# Basic CMake Setup
################################
include(cmake/CMakeBasics.cmake)

##################################
# Helpers for  VTKm device symbols
##################################
include(cmake/VTKmDeviceSymbols.cmake)

################################
# Set a default build type
################################
include(cmake/AscentBuildType.cmake)

################################
# Setup Fortran Support
################################
include(cmake/SetupFortran.cmake)

################################
# Setup 3rd Party Libs
################################
include(cmake/Setup3rdParty.cmake)

################################
# Setup tests helpers
################################
include(cmake/SetupTests.cmake)

################################
# Setup project wide includes
################################
include(cmake/SetupIncludes.cmake)

################################
# Add builtin third party libs
################################
add_subdirectory(thirdparty_builtin)

if(ENABLE_FIDES)
  find_package(ADIOS2 REQUIRED)
  find_package(Fides REQUIRED)
endif()

################################
# Add our libs
################################
add_subdirectory(flow)

if(VTKM_FOUND)
  add_subdirectory(rover)
endif()

add_subdirectory(ascent)

################################
# Add mini-app examples
################################
if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

################################
# Add our tests
################################
if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

################################
# Add utilites
################################
if(ENABLE_UTILS)
   add_subdirectory(utilities)
endif()

################################
# Add documentation targets
################################
if(ENABLE_DOCS)
  add_subdirectory(docs)
endif()

################################
# Add our config helpers
################################
add_subdirectory(config)

################################
# Create CMake importable
# exports for all of our targets
################################
if(WIN32)
    install(EXPORT ascent DESTINATION cmake)
else()
    install(EXPORT ascent DESTINATION lib/cmake/${PROJECT_NAME})
endif()
