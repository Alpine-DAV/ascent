###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


###############################################################################
#
# file: src/flow/CMakeLists.txt
#
###############################################################################

set(FLOW_PYTHON_ENABLED ${PYTHON_FOUND})

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/flow_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/flow_config.h")

set(flow_sources
    # flow interface
    flow_data.cpp
    flow_registry.cpp
    flow_filter.cpp
    flow_filters.cpp
    flow_graph.cpp
    flow_workspace.cpp
    flow_timer.cpp
    filters/flow_builtin_filters.cpp)

set(flow_headers
    flow.hpp
    flow_exports.h
    ${CMAKE_CURRENT_BINARY_DIR}/flow_config.h
    flow_data.hpp
    flow_registry.hpp
    flow_filter.hpp
    flow_filters.hpp
    flow_graph.hpp
    flow_workspace.hpp
    flow_timer.hpp
    filters/flow_builtin_filters.hpp)

set(flow_thirdparty_libs
    conduit
    conduit_relay)

#
# Flows python interpreter support enables
# running python filters when the host code
# does not provide a python interpreter
#

if(PYTHON_FOUND)
   list(APPEND flow_sources
        flow_python_interpreter.cpp
        filters/flow_python_script_filter.cpp)

   list(APPEND flow_headers
        flow_python_interpreter.hpp
        filters/flow_python_script_filter.hpp)

   # allow flow to link with python
   list(APPEND flow_thirdparty_libs python)

endif()

##########################################
# Build flow
##########################################

blt_add_library(
    NAME        ascent_flow
    SOURCES     ${flow_sources}
    HEADERS     ${flow_headers}
    DEPENDS_ON  ${flow_thirdparty_libs})

# set_target_properties(ascent_flow PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(ascent_flow PRIVATE ASCENT_EXPORTS_FLAG)
    
# install target for serial ascent lib
install(TARGETS ascent_flow
        EXPORT  ascent
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

install(FILES ${flow_headers} DESTINATION include/ascent)


################################
# Add python wrappers if python
# support was selected
################################
if(PYTHON_FOUND)
    add_subdirectory(python)
endif()


