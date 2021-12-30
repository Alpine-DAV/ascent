###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# HOLO CMake Build for Ascent
#
###############################################################################

set(HOLO_SOURCES
    holo_compare.cpp)

set(holo_deps ascent)

if(OPENMP_FOUND)
   list(APPEND deps openmp)
endif()

if (ENABLE_SERIAL)
    blt_add_executable(
        NAME        holo
        SOURCES     ${HOLO_SOURCES}
        DEPENDS_ON  ${holo_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    # install target for replay serial
    install(TARGETS holo
            EXPORT  ascent
            LIBRARY DESTINATION utilities/ascent/holo_compare
            ARCHIVE DESTINATION utilities/ascent/holo_compare
            RUNTIME DESTINATION utilities/ascent/holo_compare
    )
endif()
