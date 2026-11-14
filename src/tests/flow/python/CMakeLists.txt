###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


####################################
# Add Python Module Tests
####################################
message(STATUS "Adding flow lib python interface unit tests")

add_python_test(t_python_flow_smoke)

if(ENABLE_PYTHON_FILTERS)
    message(STATUS "Adding flow lib python script filter interface unit test")
    add_python_test(t_python_flow_inception)
endif()

