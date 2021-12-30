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
if(NOT ADIOS2_DIR)
    MESSAGE(FATAL_ERROR "ADIOS2 support needs explicit ADIOS2_DIR")
endif()

MESSAGE(STATUS "Looking for ADIOS2 using ADIOS2_DIR = ${ADIOS2_DIR}")

set(ADIOS2_DIR_ORIG ${ADIOS2_DIR})

find_package(ADIOS2 REQUIRED
             NO_DEFAULT_PATH
             PATHS ${ADIOS2_DIR}/lib/cmake/adios2)

message(STATUS "FOUND ADIOS2 at ${ADIOS2_DIR}")

blt_register_library(NAME adios2
                     INCLUDES ${ADIOS2_INCLUDE_DIR}
                     LIBRARIES ${ADIOS2_LIB_DIRS} ${ADIOS2_LIBRARIES} )
