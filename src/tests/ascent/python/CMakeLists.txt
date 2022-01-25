###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


####################################
# Add Python Module Tests
####################################
set(PYTHON_MODULE_TESTS t_python_ascent_smoke
                        t_python_ascent_render
                        t_python_ascent_flow_runtime)
set(PYTHON_MODULE_MPI_TESTS t_python_ascent_mpi_render)

message(STATUS "Adding ascent lib python interface unit tests")

if(ENABLE_SERIAL)
    foreach(TEST ${PYTHON_MODULE_TESTS})
        add_python_test(${TEST})
    endforeach()
endif()

if(MPI_FOUND)
    foreach(TEST ${PYTHON_MODULE_MPI_TESTS})
        add_python_mpi_test(${TEST} NUM_MPI_TASKS 2)
    endforeach()
endif()
