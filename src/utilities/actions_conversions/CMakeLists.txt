###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# yaml2json CMake Build for Ascent
#
###############################################################################

set(yaml2json_sources
    yaml2json.cpp)

set(yaml2json_deps conduit)

blt_add_executable(
    NAME        yaml2json
    SOURCES     ${yaml2json_sources}
    DEPENDS_ON  ${yaml2json_deps}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# install target for replay serial
install(TARGETS yaml2json
        EXPORT  ascent
        LIBRARY DESTINATION utilities/ascent/replay
        ARCHIVE DESTINATION utilities/ascent/replay
        RUNTIME DESTINATION utilities/ascent/replay
)

set(json2yaml_sources
    json2yaml.cpp)

set(json2yaml_deps conduit)

blt_add_executable(
    NAME        json2yaml
    SOURCES     ${json2yaml_sources}
    DEPENDS_ON  ${json2yaml_deps}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# install target for replay serial
install(TARGETS json2yaml
        EXPORT  ascent
        LIBRARY DESTINATION utilities/ascent/replay
        ARCHIVE DESTINATION utilities/ascent/replay
        RUNTIME DESTINATION utilities/ascent/replay
)
