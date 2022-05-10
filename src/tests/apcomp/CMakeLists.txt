###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


################################
# Unit Tests
################################

################################
# Core Unit Tests
################################
set(BASIC_TESTS t_apcomp_smoke
                t_apcomp_zbuffer
                t_apcomp_c_order
                t_apcomp_volume_partials)

set(MPI_TESTS t_apcomp_mpi_smoke
              t_apcomp_zbuffer_mpi
              t_apcomp_c_order_mpi
              t_apcomp_volume_partials_mpi)

# conduit is used to create json reports
# ascent is used for image compare
set(apcomp_deps conduit::conduit ascent)

# note: this should come in automatically
#       but not going to temp fate 
if(ENABLE_OPENMP)
  list(APPEND apcomp_deps ${ascent_blt_openmp_deps})
endif()

################################
# Add main tests
################################
if (ENABLE_SERIAL)
    message(STATUS "Adding apcomp lib unit tests")
    foreach(TEST ${BASIC_TESTS})
        add_cpp_test(TEST ${TEST} DEPENDS_ON apcomp ${apcomp_deps})
    endforeach()
endif()
################################
# Add optional tests
################################

if(MPI_FOUND AND ENABLE_MPI)
    message(STATUS "MPI enabled: Adding related apcomp unit tests")
    foreach(TEST ${MPI_TESTS})
        # this uses 4 procs
        add_cpp_mpi_test(TEST ${TEST} NUM_MPI_TASKS 4 DEPENDS_ON apcomp_mpi ${apcomp_deps})
    endforeach()
else()
    message(STATUS "MPI disabled: Skipping apcomp related tests")
endif()


# for t_utils.hpp
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
