###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


################################
# Unit Tests
################################

################################
# Flow Unit Tests
################################
set(FLOW_TESTS  t_flow_data
                t_flow_timer
                t_flow_registry
                t_flow_workspace
                t_flow_workspace_adv_manage)

if(PYTHON_FOUND)
    list(APPEND FLOW_TESTS t_flow_python_interpreter
                           t_flow_python_script_filter)
endif()

################################
# Add tests
################################

#(tests depend on ascent b/c we are using some of its utils in the tests)

# Note: flow tests use some of ascent's utils, hence the `ascent` dep here
message(STATUS "Adding flow lib unit tests")
foreach(TEST ${FLOW_TESTS})
    add_cpp_test(TEST ${TEST} DEPENDS_ON ascent ascent_flow)
endforeach()

if(PYTHON_FOUND AND ENABLE_PYTHON)
    add_subdirectory("python")
else()
    message(STATUS "Python disabled: Skipping ascent python module tests")
endif()
