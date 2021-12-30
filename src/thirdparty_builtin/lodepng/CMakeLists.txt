###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

################################
# Simple CMake setup for lodepng
################################

#
# lodepng sources
#

set(lodepng_sources
    lodepng.cpp
    )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# we always build lodepng as lib 
blt_add_library(
    NAME        ascent_lodepng
    SOURCES     ${lodepng_sources})

# install target for serial ascent lib
install(TARGETS ascent_lodepng
        EXPORT  ascent 
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)
