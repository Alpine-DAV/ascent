###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# Setup BabelFlow and ParallelMergeTree
###############################################################################


# allow prev case style "BabelFlow_DIR"
if(NOT BABELFLOW_DIR)
    if(BabelFlow_DIR)
        set(BABELFLOW_DIR ${BabelFlow_DIR})
    endif()
endif()

if(NOT BABELFLOW_DIR)
    MESSAGE(FATAL_ERROR "BabelFlow support needs explicit BABELFLOW_DIR")
endif()


MESSAGE(STATUS "Looking for BabelFlow using BABELFLOW_DIR = ${BABELFLOW_DIR}")

set(BABELFLOW_DIR_ORIG ${BABELFLOW_DIR})

find_package(BabelFlow REQUIRED
             NO_DEFAULT_PATH
             PATHS ${BABELFLOW_DIR}/lib/cmake/)

message(STATUS "FOUND BabelFlow at ${BABELFLOW_DIR}")

set(BABELFLOW_FOUND TRUE)

blt_register_library( NAME babelflow
                      INCLUDES ${BabelFlow_INCLUDE_DIRS}
                      LIBRARIES  babelflow babelflow_mpi)

## Find also ParallelMergeTree analysis algorithm to build (based on BabelFlow)
if(NOT PMT_DIR)
    MESSAGE(FATAL_ERROR "ParallelMergeTree support needs explicit PMT_DIR")
endif()

MESSAGE(STATUS "Looking for ParallelMergeTree using PMT_DIR = ${PMT_DIR}")

set(PMT_DIR_ORIG ${PMT_DIR})

find_package(PMT REQUIRED
             NO_DEFAULT_PATH
             PATHS ${PMT_DIR}/lib/cmake)

message(STATUS "FOUND PMT at ${PMT_DIR}")

blt_register_library( NAME pmt
                      INCLUDES ${PMT_INCLUDE_DIRS}
                      LIBRARIES  pmt)
