###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


# this header allows us to easily use the cmake source and binary paths in
# our unit tests
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/t_config.hpp.in"
                "${CMAKE_CURRENT_BINARY_DIR}/t_config.hpp")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

# for t_utils.hpp
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

################################
# Unit Tests
################################
# add tpl smoke tests, so they run before we run the ascent tests
add_subdirectory("thirdparty")

# add apcomp tests
if(ENABLE_APCOMP)
    add_subdirectory("apcomp")
endif()

# add flow tests
add_subdirectory("flow")

# add ascent tests
add_subdirectory("ascent")

# add ascent performance tests
add_subdirectory("perf")

if(PYTHON_FOUND)
# if we have python:
#  add custom command that generates a html img comparison report for our tests
#  run with "make gen_report"
#  creates "_output/tout_img_report.html"
    add_custom_target(gen_report
                      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_img_compare_html_report.py
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      COMMENT "Generating html summary of image compare tests (tests/_output/tout_img_report.html)")
endif()

