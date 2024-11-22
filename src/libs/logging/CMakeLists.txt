# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.


# gen config header
set(ASCENT_LOGGING_ENABLE_CALIPER ${CALIPER_FOUND})
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/ascent_logging_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/ascent_logging_config.h")

set(ascent_logging_headers
    ${CMAKE_CURRENT_BINARY_DIR}/ascent_logging_config.h
    ascent_logging.hpp
    ascent_logging_timer.hpp
    ascent_annotations.hpp
    ascent_logging_exports.h
  )

set(ascent_logging_sources
    ascent_logging.cpp
    ascent_logging_timer.cpp
    ascent_annotations.cpp
  )


set(ascent_logging_deps conduit::conduit)

if(CALIPER_FOUND)
    list(APPEND ascent_logging_deps caliper)
endif()

blt_add_library(NAME ascent_logging
                SOURCES ${ascent_logging_sources}
                HEADERS ${aascent_logging_headers}
                DEPENDS_ON ${ascent_logging_deps})

# for generated config header
target_include_directories(ascent_logging PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
target_include_directories(ascent_logging PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/logging>)

# extra defs and props
target_compile_definitions(ascent_logging PRIVATE ASCENT_EXPORTS_FLAG)

if(ENABLE_HIDDEN_VISIBILITY)
    set_target_properties(ascent_logging PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

install(FILES ${ascent_logging_headers} DESTINATION include/ascent/)

install(TARGETS ascent_logging
        EXPORT  ascent
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
        )
