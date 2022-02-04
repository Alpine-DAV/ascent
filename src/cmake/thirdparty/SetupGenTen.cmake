###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


###############################################################################
# Setup GenTen
###############################################################################

if(NOT GENTEN_DIR)
  MESSAGE(FATAL_ERROR "GenTen support needs explicit GENTEN_DIR")
endif()

MESSAGE(STATUS "Looking for GenTen using GENTEN_DIR = ${GENTEN_DIR}")


set(GENTEN_DIR_ORIG ${GENTEN_DIR})

set(Genten_DIR ${GENTEN_DIR}/lib64/cmake/Genten/)

find_package(Genten REQUIRED QUIET)

set(GENTEN_FOUND TRUE)
set(GENTEN_INCLUDE_DIR ${GENTEN_DIR}/include/genten/)

blt_register_library(NAME genten
                     LIBRARIES gt_higher_moments
                     INCLUDES ${GENTEN_INCLUDE_DIR}
                     )
