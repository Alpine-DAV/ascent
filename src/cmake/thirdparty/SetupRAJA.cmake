# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

message(STATUS "Looking for RAJA in: ${RAJA_DIR}")
if (NOT RAJA_DIR)
  message(FATAL_ERROR "Must specify 'RAJA_DIR'")
endif()

set(RAJA_DIR_ORIG ${RAJA_DIR})
find_dependency(RAJA REQUIRED
                NO_DEFAULT_PATH
                PATHS ${RAJA_DIR}/share/raja/cmake)

message(STATUS "Found RAJA")
