###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# Setup Kokkos
###############################################################################
if(KOKKOS_DIR)
    MESSAGE("PROVIDED KOKKOS DIR:${KOKKOS_DIR}")

    # check for both lib64 and lib
    if(EXISTS ${KOKKOS_DIR}/lib64/cmake/Kokkos/)
        set(KOKKOS_CMAKE_CONFIG_DIR ${KOKKOS_DIR}/lib64/cmake/Kokkos/)
    endif()

    if(EXISTS ${KOKKOS_DIR}/lib/cmake/Kokkos/)
        set(KOKKOS_CMAKE_CONFIG_DIR ${KOKKOS_DIR}/lib/cmake/Kokkos/)
    endif()

    if(NOT EXISTS ${KOKKOS_CMAKE_CONFIG_DIR}/KokkosConfig.cmake)
        MESSAGE(FATAL_ERROR "Could not find kokkos CMake include file (${KOKKOS_CMAKE_CONFIG_DIR}/KokkosConfig.cmake)")
    endif()

    ###############################################################################
    # Import Kokkos CMake targets
    ###############################################################################
    find_package(Kokkos REQUIRED
                 NO_DEFAULT_PATH
                 COMPONENTS separable_compilation
                 PATHS ${KOKKOS_CMAKE_CONFIG_DIR})
         MESSAGE("FOUND KOKKOS PACKAGE:${KOKKOS_CMAKE_CONFIG_DIR}")
endif()

