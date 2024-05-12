###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


####################################
# Add Util Tests
####################################

if(PYTHON_FOUND AND ENABLE_PYTHON)
    message(STATUS "Adding ascent utility tests")
    add_python_test(t_python_visit_session_converters)
else()
    message(STATUS "Python disabled: Skipping ascent python utility tests")
endif()


