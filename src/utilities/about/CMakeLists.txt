###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# About Util CMake Build for Ascent
#
###############################################################################

set(ABOUT_SOURCES
    about.cpp)

set(ABOUT_DEPS ascent)

if(OPENMP_FOUND)
   list(APPEND deps openmp)
endif()

if (ENABLE_SERIAL)
    blt_add_executable(
        NAME        ascent_about
        SOURCES     ${ABOUT_SOURCES}
        DEPENDS_ON  ${ABOUT_DEPS}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    
    # install target for replay serial
    install(TARGETS ascent_about
            EXPORT  ascent
            LIBRARY DESTINATION utilities/ascent/
            ARCHIVE DESTINATION utilities/ascent/
            RUNTIME DESTINATION utilities/ascent/
    )
endif()

if(MPI_FOUND)

    set(ABOUT_MPI_DEPS ascent_mpi mpi)
    if(OPENMP_FOUND)
           list(APPEND about_mpi_deps openmp)
    endif()

    blt_add_executable(
        NAME        ascent_about_mpi
        SOURCES     ${ABOUT_SOURCES}
        DEPENDS_ON  ${ABOUT_MPI_DEPS}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    blt_add_target_compile_flags(TO ascent_about_mpi FLAGS "-DABOUT_MPI=1")

    # install target for replay mpi
    install(TARGETS ascent_about_mpi
            EXPORT  ascent
            LIBRARY DESTINATION utilities/ascent/
            ARCHIVE DESTINATION utilities/ascent/
            RUNTIME DESTINATION utilities/ascent/
    )

endif()






