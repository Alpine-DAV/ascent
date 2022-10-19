###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


################################
#  Project Wide Includes
################################

# add lodepng include dir
include_directories(${PROJECT_SOURCE_DIR}/thirdparty_builtin/lodepng)

# add tiny_obj_loader include dir
include_directories(${PROJECT_SOURCE_DIR}/thirdparty_builtin/tiny_obj)

#############
# TODO: Fix these with build vs install interface target includes
#############

# add include dirs so units tests have access to the headers across
# libs and in unit tests
include_directories(${PROJECT_SOURCE_DIR}/ascent/)
include_directories(${PROJECT_SOURCE_DIR}/ascent/c)
include_directories(${PROJECT_BINARY_DIR}/ascent/)
include_directories(${PROJECT_SOURCE_DIR}/ascent/utils)
include_directories(${PROJECT_SOURCE_DIR}/ascent/runtimes)
include_directories(${PROJECT_SOURCE_DIR}/ascent/hola)
include_directories(${PROJECT_SOURCE_DIR}/ascent/runtimes/flow_filters)

