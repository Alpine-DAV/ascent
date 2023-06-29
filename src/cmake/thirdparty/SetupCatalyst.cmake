###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# Setup VTKm
###############################################################################

if(NOT CATALYST_DIR)
  MESSAGE(FATAL_ERROR "CATALYST support needs explicit CATALYST_DIR")
endif()

MESSAGE(STATUS "Looking for CATALYST 2.0 using CATALYST_DIR = ${CATALYST_DIR}")

# use CATALYST_DIR to setup the options that cmake's find Catalyst needs
file(GLOB Catalyst_DIR "${CATALYST_DIR}")
if(NOT Catalyst_DIR)
    MESSAGE(FATAL_ERROR "Failed to find CATALYST at CATALYST_DIR=${CATALYST_DIR}")
endif()

find_package(Catalyst REQUIRED COMPONENTS SDK)

set(CATALYST_FOUND TRUE)
