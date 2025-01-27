###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


################################
# Unit Tests
################################
set(BASIC_TESTS t_ascent_logger_basic)
# set(MPI_TESTS   t_ascent_logger_mpi)


message(STATUS "Adding logger unit tests")
foreach(TEST ${BASIC_TESTS})
    message(STATUS "adding test ${TEST}")
    add_cpp_test(TEST ${TEST} DEPENDS_ON ascent_logging)
endforeach()

# if(MPI_FOUND)
#     message(STATUS "Adding logger mpi unit tests")
#     foreach(TEST ${MPI_TESTS})
#         message(STATUS "adding MPI test ${TEST}")
#         add_cpp_mpi_test(TEST ${TEST} NUM_MPI_TASKS 2 DEPENDS_ON ascent_logging)
#     endforeach()
# endif()
