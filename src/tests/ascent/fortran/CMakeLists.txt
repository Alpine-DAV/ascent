###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


###############################################################################
# Ascent Fortran Interface Unit Tests
###############################################################################
if (ENABLE_SERIAL)
    set(FORTRAN_TESTS
        t_f_ascent_smoke
        t_f_ascent_render_2d)
    
    
    ################################
    # Add our tests
    ################################
    message(STATUS "Adding ascent lib fortran interface unit tests")
    foreach(TEST ${FORTRAN_TESTS})
        add_fortran_test(TEST ${TEST} DEPENDS_ON ascent
                                                 conduit_blueprint)
    endforeach()
endif()



