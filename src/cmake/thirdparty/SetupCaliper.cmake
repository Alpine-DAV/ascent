# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################
#
# Setup CALIPER


# first Check for CALIPER_DIR

if(NOT CALIPER_DIR)
    MESSAGE(FATAL_ERROR "Caliper support needs explicit CALIPER_DIR")
endif()


# first: look for caliper config header + see what additional deps we need
#.       to resolve.

message(STATUS "Attempting to find cali-config.h with CALIPER_DIR=${CALIPER_DIR} ...")
find_file(CALI_CONFIG_HEADER
          NAMES caliper-config.h
          PATHS ${CALIPER_DIR}
          PATH_SUFFIXES include/caliper
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

if(EXISTS ${CALI_CONFIG_HEADER})
    message(STATUS "Found Caliper Config Header: ${CALI_CONFIG_HEADER}")
else()
    message(FATAL_ERROR "Could not find caliper-config.h in caliper ${CALIPER_DIR}/include/caliper")
endif()

file(READ ${CALI_CONFIG_HEADER} _CALI_CONFIG_HEADER_CONTENTS)

# check if we need ADIAK
string(FIND  ${_CALI_CONFIG_HEADER_CONTENTS} "#define CALIPER_HAVE_ADIAK" _caliper_have_adiak)

if(${_caliper_have_adiak} GREATER_EQUAL 0 )
    # caliper is built with adiak support and caliper needs us to find adiak,
    # else find_pacakge caliper will fail
    # Check for ADIAK_DIR
    if(NOT ADIAK_DIR)
        MESSAGE(FATAL_ERROR "Caliper support needs explicit ADIAK_DIR")
    endif()
    # find adiak
    find_package(adiak REQUIRED
                 NO_DEFAULT_PATH
                 PATHS ${ADIAK_DIR}/lib/cmake/adiak)
    set(ADIAK_FOUND TRUE)
endif()


find_package(caliper REQUIRED
             NO_DEFAULT_PATH
             PATHS ${CALIPER_DIR}/share/cmake/caliper)


set(CALIPER_FOUND TRUE)
