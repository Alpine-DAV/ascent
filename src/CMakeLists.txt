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


cmake_minimum_required(VERSION 3.0)
################################
# cmake policy selections
################################
# avoid default mac osx rpath settings for cmake 3.0
cmake_policy(SET CMP0042 OLD)
# avoid warnings for project commands w/o VERSION
cmake_policy(SET CMP0048 NEW)


################################
# Alpine
################################

project(alpine VERSION "0.1.0")

################################
# Build Options
################################
option(BUILD_SHARED_LIBS  "Build shared libraries"    ON)
option(ENABLE_TESTS       "Build tests"               ON)

option(ENABLE_FORTRAN     "Build Fortran support"     ON)
option(ENABLE_PYTHON      "Build Python Support"      ON)

option(ENABLE_MPI         "Build MPI Support"         ON)
option(ENABLE_CUDA        "Build CUDA Support"        OFF)
option(ENABLE_OPENMP      "Build OpenMP Support"      OFF)

################################
# Invoke CMake Fortran setup
# if ENABLE_FORTRAN == ON
################################
if(ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

include(blt/SetupBLT.cmake)

################################
# Basic CMake Setup
################################
include(cmake/CMakeBasics.cmake)

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

################################
# Add our libs
################################
add_subdirectory(alpine)

################################
# Add vtk-h 
################################
if(VTKM_FOUND)
    add_subdirectory(vtk-h)
endif()

################################
# Add our tests
################################
if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

################################
# Add mini-app examples
################################
add_subdirectory(examples)


################################
# Add documentation targets
################################
add_subdirectory(docs)


################################
# Create CMake importable 
# exports for all of our targets
################################
install(EXPORT alpine DESTINATION lib/cmake)





