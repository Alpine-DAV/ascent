# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

set(color_tables_thirdparty_libs dray)


################################################
# openmp support
################################################
if(ENABLE_OPENMP)
    list(APPEND color_tables_thirdparty_libs ${ascent_blt_openmp_deps})
endif()

if(ENABLE_SERIAL)
    blt_add_executable(NAME color_tables
                       SOURCES color_tables.cpp
                       DEPENDS_ON ${color_tables_thirdparty_libs}
                       OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                       )
else()
    message(STATUS "Serial disabled: Skipping color tables utility exe")
endif()
