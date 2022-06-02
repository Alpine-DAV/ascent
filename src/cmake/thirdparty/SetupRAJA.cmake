# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

if (NOT RAJA_DIR)
  message(FATAL_ERROR "RAJA support needs explicit RAJA_DIR")
endif()

message(STATUS "Looking for RAJA in: ${RAJA_DIR}")

set(RAJA_DIR_ORIG ${RAJA_DIR})
find_dependency(RAJA REQUIRED
                NO_DEFAULT_PATH
                PATHS ${RAJA_DIR}/share/raja/cmake)
message(STATUS "Found RAJA")
