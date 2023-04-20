# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

if(NOT CAMP_DIR)
  message(FATAL_ERROR "Camp support needs explicit CAMP_DIR")
endif()

message(STATUS "Looking for Camp in: ${CAMP_DIR}")

set(_CAMP_SEARCH_PATH)
if(EXISTS  ${CAMP_DIR}/share/camp/cmake)
  # old install layout ?
  set(_CAMP_SEARCH_PATH ${CAMP_DIR}/share/camp/cmake)
else()
  # new install layout ?
  set(_CAMP_SEARCH_PATH ${CAMP_DIR}/lib/cmake/camp)
endif()

set(CAMP_DIR_ORIG ${CAMP_DIR})
find_package(camp REQUIRED
             NO_DEFAULT_PATH
             PATHS ${_CAMP_SEARCH_PATH})
             
message(STATUS "Found Camp in: ${CAMP_DIR}")
set(CAMP_FOUND TRUE)

if(ASCENT_ENABLE_TESTS AND WIN32 AND BUILD_SHARED_LIBS)
    # if we are running tests with dlls, we need path to dlls
    list(APPEND ASCENT_TPL_DLL_PATHS ${CAMP_DIR_ORIG}/lib/)
endif()