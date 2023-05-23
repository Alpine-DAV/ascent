# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

if(NOT UMPIRE_DIR)
  message(FATAL_ERROR "Umpire support needs explicit UMPIRE_DIR")
endif()

message(STATUS "Looking for Umpire in: ${UMPIRE_DIR}")

set(_UMPIRE_SEARCH_PATH)
if(EXISTS ${UMPIRE_DIR}/share/umpire/cmake)
  # old install layout
  set(_UMPIRE_SEARCH_PATH ${UMPIRE_DIR}/share/umpire/cmake)
else()
  # new install layout
  set(_UMPIRE_SEARCH_PATH ${UMPIRE_DIR}/lib/cmake/umpire)
endif()

set(UMPIRE_DIR_ORIG ${UMPIRE_DIR})
find_package(umpire REQUIRED
             NO_DEFAULT_PATH
             PATHS ${_UMPIRE_SEARCH_PATH})

message(STATUS "Found Umpire in: ${UMPIRE_DIR}")
set(UMPIRE_FOUND TRUE)

if(ASCENT_ENABLE_TESTS AND WIN32 AND BUILD_SHARED_LIBS)
    # if we are running tests with dlls, we need path to dlls
    list(APPEND ASCENT_TPL_DLL_PATHS ${UMPIRE_DIR_ORIG}/lib/)
endif()