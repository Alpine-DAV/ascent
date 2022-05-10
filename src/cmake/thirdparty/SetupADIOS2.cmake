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

# Handle legacy usage of ADIOS2_DIR
if (ADIOS2_DIR AND NOT ADIOS2_ROOT)
  # If find_package(ADIOS2) has already been called this will fail
  if (NOT EXISTS ${ADIOS2_DIR}/include)
    get_filename_component(tmp "${ADIOS2_DIR}" DIRECTORY)
    get_filename_component(tmp "${tmp}" DIRECTORY)
    get_filename_component(tmp "${tmp}" DIRECTORY)
    if (EXISTS ${tmp}/include)
      set(ADIOS2_ROOT "${tmp}" CACHE PATH "")
    else ()
      message(FATAL_ERROR "Could not determine ADIOS2_ROOT from ADIOS2_DIR")
    endif ()
  else ()
    set(ADIOS2_ROOT "${ADIOS2_DIR}" CACHE PATH "")
  endif ()
endif ()

# Check for ADIOS_ROOT
if(NOT ADIOS2_ROOT)
    MESSAGE(FATAL_ERROR "ADIOS2 support needs explicit ADIOS2_ROOT")
endif()

MESSAGE(STATUS "Looking for ADIOS2 using ADIOS2_ROOT = ${ADIOS2_ROOT}")

set(ADIOS2_DIR_ORIG ${ADIOS2_ROOT})

find_package(ADIOS2 REQUIRED
             NO_DEFAULT_PATH
             PATHS ${ADIOS2_ROOT})

# ADIOS2_DIR is set by find_package
message(STATUS "FOUND ADIOS2 at ${ADIOS2_DIR}")

blt_register_library(NAME adios2
  INCLUDES ${ADIOS2_INCLUDE_DIR}
  LIBRARIES ${ADIOS2_LIB_DIRS} ${ADIOS2_LIBRARIES} )
