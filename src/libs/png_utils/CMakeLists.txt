# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

set(ascent_png_utils_headers
    ascent_png_compare.hpp
    ascent_png_decoder.hpp
    ascent_png_encoder.hpp
    ascent_png_utils_exports.h
  )

set(ascent_png_utils_sources
    ascent_png_compare.cpp
    ascent_png_decoder.cpp
    ascent_png_encoder.cpp
  )

install(FILES ${ascent_png_utils_headers} DESTINATION include/ascent/png_utils)

set(ascent_png_utils_deps conduit::conduit ascent_lodepng)

if(ENABLE_OPENMP)
    list(APPEND ascent_png_utils_deps ${ascent_blt_openmp_deps})
endif()


blt_add_library(NAME ascent_png_utils
                OBJECT TRUE
                SOURCES ${ascent_png_utils_sources}
                HEADERS ${ascent_png_utils_headers}
                DEPENDS_ON ${ascent_png_utils_deps})

target_include_directories(ascent_png_utils PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)

# extra defs and props
target_compile_definitions(ascent_png_utils PRIVATE ASCENT_EXPORTS_FLAG)

if(ENABLE_HIDDEN_VISIBILITY)
    set_target_properties(ascent_png_utils PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

