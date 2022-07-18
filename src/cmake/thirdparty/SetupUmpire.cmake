# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

if(NOT UMPIRE_DIR)
  message(FATAL_ERROR "Umpire support needs explicit UMPIRE_DIR")
endif()

message(STATUS "Looking for Umpire in: ${UMPIRE_DIR}")

set(_UMPIRE_SEARCH_PATH)
if(EXISTS  ${UMPIRE_DIR}/share/umpire/cmake)
  # old umpire install layout
  set(_UMPIRE_SEARCH_PATH ${UMPIRE_DIR}/share/umpire/cmake)
else()
  # new umpire install layout
  set(_UMPIRE_SEARCH_PATH ${UMPIRE_DIR}/lib/cmake/umpire)
endif()

set(UMPIRE_DIR_ORIG ${UMPIRE_DIR})
find_package(umpire REQUIRED
             NO_DEFAULT_PATH
             PATHS ${_UMPIRE_SEARCH_PATH})
             
message(STATUS "Found Umpire in: ${UMPIRE_DIR}")
set(UMPIRE_FOUND TRUE)
