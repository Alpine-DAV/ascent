# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

if(NOT OCCA_DIR)
  MESSAGE(FATAL_ERROR "Occa support needs explicit OCCA_DIR")
endif()

MESSAGE(STATUS "Looking for OCCA using OCCA_DIR = ${OCCA_DIR}")

set(occa_DIR ${OCCA_DIR}/lib)
message(STATUS "Found OCCA include dirs: ${OCCA_INCLUDE_DIRS}")

find_path(OCCA_INCLUDE_DIRS occa.hpp
          PATHS ${OCCA_DIR}/include
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

#find libs
find_library(OCCA_LIBRARIES LIBRARIES NAMES occa
             PATHS ${OCCA_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

# also search for cudatoolkit
if(ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND OCCA_LIBRARIES CUDA::cuda_driver)
endif()

blt_register_library(NAME occa
                     INCLUDES ${OCCA_INCLUDE_DIRS}
                     LIBRARIES ${OCCA_LIBRARIES})

set(OCCA_FOUND TRUE)
