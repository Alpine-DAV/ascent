###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# Setup ANARI
###############################################################################

if(NOT ANARI_DIR)
  MESSAGE(FATAL_ERROR "ANARI support needs explicit ANARI_DIR")
endif()

MESSAGE(STATUS "Looking for ANARI using ANARI_DIR = ${ANARI_DIR}")

set(ANARI_DIR_ORIG ${ANARI_DIR})
set(ANARI_FOUND TRUE)

file(GLOB ANARI_DIR "${ANARI_DIR}/lib/cmake/anari-*")
if(NOT EXISTS ${ANARI_DIR}/anariConfig.cmake)
    MESSAGE(FATAL_ERROR "Could not find ANARI CMake at (${ANARI_DIR}/lib/cmake/anari-*)")
endif()

###############################################################################
# Import ANARI CMake targets
###############################################################################
#find_package(anari REQUIRED)
find_package(anari REQUIRED
         NO_DEFAULT_PATH
         PATHS ${ANARI_DIR})
if(NOT TARGET vtkm::anari)
    message(FATAL_ERROR "vtkm::anari not found, check your VTK-m install")
endif()

