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

################################
# Alpine 3rd Party Dependencies
################################

if(ENABLE_TESTS)
    add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
    ################################
    # Enable GTest
    ################################
    
    #
    # We always want to build gtest as a static lib, however
    # it shares our "BUILD_SHARED_LIBS" option, so we need
    # to force this value to OFF, and then restore the 
    # previous setting.
    #

    set(BSL_ORIG_VALUE ${BUILD_SHARED_LIBS})
    
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(thirdparty_builtin/gtest-1.7.0)
    
    set(BUILD_SHARED_LIBS ${BSL_ORIG_VALUE})
    
    enable_testing()
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

endif()

################################
# Setup Python if requested
################################
if(ENABLE_PYTHON)
    include(CMake/thirdparty/SetupPython.cmake)
    message(STATUS "Using Python Include: ${PYTHON_INCLUDE_DIRS}")
    include_directories(${PYTHON_INCLUDE_DIRS})
    # if we don't find python, throw a fatal error
    if(NOT PYTHON_FOUND)
        message(FATAL_ERROR "ENABLE_PYTHON is true, but Python wasn't found.")
    endif()
endif()

################################
# Conduit
################################
include(CMake/thirdparty/SetupConduit.cmake)


################################################################
################################################################
#
# 3rd Party Libs that underpin Alpine's Pipelines
#
################################################################
################################################################

################################
# Make sure we have a concrete
# pipeline to build 
################################
<<<<<<< HEAD
if(NOT EAVL_DIR AND NOT VTKM_DIR AND NOT HDF5_DIR AND NOT ADIOS_DIR)
    message(FATAL_ERROR "Strawman requires at least once concrete pipeline (EAVL or VTKm)")
endif()



################################
# EAVL and supporting libs
################################
if(EAVL_DIR)

     # OSMesa
    ################################
    include(CMake/thirdparty/SetupOSMesa.cmake)
    ################################
    # EAVL
    ################################
    include(CMake/thirdparty/SetupEAVL.cmake)
   
   
endif()

=======
if(NOT VTKM_DIR AND NOT HDF5_DIR)
    message(FATAL_ERROR "Alpine requires at least once concrete pipeline (EAVL for VTKm)")
endif()


>>>>>>> c86fd9e32d8eb7b1d46bd439503701dc527a1188
################################
# VTKm and supporting libs
################################
if(VTKM_DIR)
    # explicitly setting this avoids a bug with VTKm's cuda
    # arch detection logic
    set(VTKm_CUDA_Architecture "kepler" CACHE PATH "" FORCE)

    ################################
    # TBB (for VTK-M)
    ################################
    message(STATUS "If VTK-m was configured with TBB then you must specify the TBB_DIR")
    if(TBB_DIR) # optional 
        include(CMake/thirdparty/SetupTBB.cmake)
    endif()

    ################################
    # VTKm
    ################################
    include(CMake/thirdparty/SetupVTKm.cmake)
endif()


################################
# Setup HDF5
################################
if(HDF5_DIR)
    include(CMake/thirdparty/SetupHDF5.cmake)
endif()

if(ADIOS_DIR)
    include(CMake/thirdparty/SetupADIOS.cmake)
    #include(CMake/thirdparty/SetupVTKm.cmake)
endif()


################################
# Optional Features
################################

################################
# Documentation Packages
################################
# Doxygen
find_package(Doxygen)
# Sphinx
include(CMake/thirdparty/FindSphinx.cmake)

################################
# Setup MPI if available 
################################
# Search for MPI.
if(ENABLE_MPI)
    include(FindMPI)
    # if we don't find mpi, throw a fatal error
    if(NOT MPI_FOUND)
        message(FATAL_ERROR "ENABLE_MPI is true, but MPI wasn't found.")
    endif()
    
    # Fortran: check for mpi module vs mpi header
    if(ENABLE_FORTRAN)
        find_path(mpif_path 
                  NAMES mpif.h
                  PATHS ${MPI_Fortran_INCLUDE_PATH} 
                  NO_DEFAULT_PATH)
        if(${mpif_path})
            set(MPI_Fortran_USE_MODULE OFF CACHE PATH "")
            message(STATUS "Found MPI_Fortran: mpif.h")
        else()
            set(MPI_Fortran_USE_MODULE ON CACHE PATH "")
            message(STATUS "Found MPI_Fortran: mpi.mod")
        endif()

    endif()    
endif()

################################
# IceT
################################
if(ENABLE_MPI)
    include(CMake/thirdparty/SetupIceT.cmake)
endif()


################################
# Setup fruit (fortran unit testing framework) if fortran is enabled
################################
if(ENABLE_FORTRAN)
    add_subdirectory(thirdparty_builtin/fruit-3.3.9)
endif()

