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
elseif(EXISTS ${UMPIRE_DIR}/lib/cmake/umpire)
  # new install layout
  set(_UMPIRE_SEARCH_PATH ${UMPIRE_DIR}/lib/cmake/umpire)
elseif(EXISTS ${UMPIRE_DIR}/lib64/cmake/umpire)
  # new install layout
  set(_UMPIRE_SEARCH_PATH ${UMPIRE_DIR}/lib64/cmake/umpire)
endif()

set(UMPIRE_DIR_ORIG ${UMPIRE_DIR})
find_package(umpire REQUIRED
             NO_DEFAULT_PATH
             PATHS ${_UMPIRE_SEARCH_PATH})

message(STATUS "Found Umpire in: ${UMPIRE_DIR}")
set(UMPIRE_FOUND TRUE)

if(ASCENT_ENABLE_TESTS AND WIN32 AND BUILD_SHARED_LIBS)
    # if we are running tests with dlls, we need path to dlls
    # hey, now we have to look at bin for the dlls :-(
    if(EXISTS ${UMPIRE_DIR_ORIG}/bin/)
        list(APPEND ASCENT_TPL_DLL_PATHS ${UMPIRE_DIR_ORIG}/bin/)
    endif()
    
    if(EXISTS ${UMPIRE_DIR_ORIG}/lib)
        list(APPEND ASCENT_TPL_DLL_PATHS ${UMPIRE_DIR_ORIG}/lib/)
    elseif(EXISTS ${UMPIRE_DIR_ORIG}/lib64) 
        # lib64 shouldn't happen on windows, but someone might
        # be clever and suprise us
        list(APPEND ASCENT_TPL_DLL_PATHS ${UMPIRE_DIR_ORIG}/lib64/)
    endif()
endif()