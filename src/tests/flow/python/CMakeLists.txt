###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


####################################
# Add Python Module Tests
####################################
set(PYTHON_MODULE_TESTS t_python_flow_smoke t_python_flow_inception)

message(STATUS "Adding flow lib python interface unit tests")
foreach(TEST ${PYTHON_MODULE_TESTS})
    add_python_test(${TEST})
endforeach()

