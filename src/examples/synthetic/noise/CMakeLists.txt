###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# Noise CMake Build for Ascent
#
###############################################################################

set(noise_sources
    noise.cpp
    open_simplex_noise.c)

set(noise_header
    open_simplex_noise.h)

set(noise_deps ascent)
if(OPENMP_FOUND)
   set(noise_openmp_flags "-DNOISE_USE_OPENMP")
   list(APPEND noise_deps openmp)
else()
   set(noise_openmp_flags "")
endif()

if (ENABLE_SERIAL)
    blt_add_executable(
        NAME        noise_ser
        SOURCES     ${noise_sources}
        DEPENDS_ON  ${noise_deps}
        OUTPUT_DIR  ${CMAKE_CURRENT_BINARY_DIR})
    
    blt_add_target_compile_flags(TO noise_ser FLAGS "${noise_openmp_flags}")
    
    # install target for noise example
    install(TARGETS noise_ser
            EXPORT  ascent
            LIBRARY DESTINATION examples/ascent/synthetic/noise
            ARCHIVE DESTINATION examples/ascent/synthetic/noise
            RUNTIME DESTINATION examples/ascent/synthetic/noise
    )
endif()

install(FILES ASCENT_README.md DESTINATION examples/ascent/synthetic/noise)

if(MPI_FOUND)

    set(noise_par_deps ascent_mpi mpi)
    if(OPENMP_FOUND)
           list(APPEND noise_par_deps openmp)
    endif()

    blt_add_executable(
        NAME        noise_par
        SOURCES     ${noise_sources}
        DEPENDS_ON  ${noise_par_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    blt_add_target_compile_flags(TO noise_par FLAGS "-DPARALLEL  ${noise_openmp_flags}")

    # install target for mpi noise example
    install(TARGETS noise_par
            EXPORT  ascent
            LIBRARY DESTINATION examples/ascent/synthetic/noise
            ARCHIVE DESTINATION examples/ascent/synthetic/noise
            RUNTIME DESTINATION examples/ascent/synthetic/noise
    )


endif()


configure_file(example_actions.yaml ${CMAKE_CURRENT_BINARY_DIR}/example_actions.yaml COPYONLY)




