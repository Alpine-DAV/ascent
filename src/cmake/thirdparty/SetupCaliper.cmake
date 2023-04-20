# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
###############################################################################
#
# Setup CALIPER


# first Check for CALIPER_DIR

if(NOT CALIPER_DIR)
    MESSAGE(FATAL_ERROR "Caliper support needs explicit CALIPER_DIR")
endif()

# most common case: caliper is built with adiak support
# and caliper needs us to find adiak, or else find_pacakge caliper
# will fail

# Check for ADIAK_DIR

if(NOT ADIAK_DIR)
    MESSAGE(FATAL_ERROR "Caliper support needs explicit ADIAK_DIR")
endif()

message(STATUS "Looking for Adiak in: ${ADIAK_DIR}")

find_package(adiak REQUIRED
             NO_DEFAULT_PATH
             PATHS ${ADIAK_DIR}/lib/cmake/adiak)

message(STATUS "Looking for Caliper in: ${CALIPER_DIR}")

find_package(caliper REQUIRED
             NO_DEFAULT_PATH
             PATHS ${CALIPER_DIR}/share/cmake/caliper)

set(ADIAK_FOUND TRUE)
set(CALIPER_FOUND TRUE)
set(ASCENT_CALIPER_ENABLED TRUE)

