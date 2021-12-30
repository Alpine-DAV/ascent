###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# Setup ADIOS
#
###############################################################################

# first Check for ADIOS_DIR

if(NOT ADIOS_DIR)
    MESSAGE(FATAL_ERROR "ADIOS support needs explicit ADIOS_DIR")
endif()

MESSAGE(STATUS "Looking for ADIOS using ADIOS_DIR = ${ADIOS_DIR}")

# CMake's FindADIOS module uses the ADIOS_ROOT env var
set(ADIOS_ROOT ${ADIOS_DIR})
set(ENV{ADIOS_ROOT} ${ADIOS_ROOT})

# Use CMake's FindADIOS module, which uses hdf5's compiler wrappers to extract
# all the info about the hdf5 install
include(${ADIOS_DIR}/etc/FindADIOS.cmake)

# FindADIOS sets ADIOS_DIR to it's installed CMake info if it exists
# we want to keep ADIOS_DIR as the root dir of the install to be
# consistent with other packages

set(ADIOS_DIR ${ADIOS_ROOT} CACHE PATH "" FORCE)
# not sure why we need to set this, but we do
#set(ADIOS_FOUND TRUE CACHE PATH "" FORCE)

if(NOT ADIOS_FOUND)
    message(FATAL_ERROR "ADIOS_DIR is not a path to a valid ADIOS install")
endif()

blt_register_library(NAME adios
                     INCLUDES ${ADIOS_INCLUDE_DIRS}
                     LIBRARIES ${ADIOS_LIBRARIES} )

