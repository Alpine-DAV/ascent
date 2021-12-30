###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# LULESH CMake Build for Ascent
#
###############################################################################

set(LULESH_SOURCES
    lulesh.cc
    lulesh-comm.cc
    lulesh-viz.cc
    lulesh-util.cc
    lulesh-init.cc)

configure_file(ascent_actions.yaml ${CMAKE_CURRENT_BINARY_DIR}/ascent_actions.yaml COPYONLY)
configure_file(ascent_options.yaml ${CMAKE_CURRENT_BINARY_DIR}/ascent_options.yaml COPYONLY)

set(lulesh_deps ascent)

if(OPENMP_FOUND)
   set(lulesh_openmp_flags "-DLULESH_USE_OPENMP")
   list(APPEND lulesh_deps openmp)
else()
   set(lulesh_openmp_flags "")
endif()

if (ENABLE_SERIAL)
    blt_add_executable(
        NAME        lulesh_ser
        SOURCES     ${LULESH_SOURCES}
        DEPENDS_ON  ${lulesh_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    
    
    blt_add_target_compile_flags(TO lulesh_ser FLAGS "-DUSE_MPI=0 ${lulesh_openmp_flags}")
    
    # install target for lulesh serial
    install(TARGETS lulesh_ser
            EXPORT  ascent
            LIBRARY DESTINATION examples/ascent/proxies/lulesh
            ARCHIVE DESTINATION examples/ascent/proxies/lulesh
            RUNTIME DESTINATION examples/ascent/proxies/lulesh
    )
endif()

install(FILES ASCENT_README.md
              ascent_actions.json
              ascent_options.json
        DESTINATION examples/ascent/proxies/lulesh)


if(MPI_FOUND)

    set(lulesh_par_deps ascent_mpi mpi)
    if(OPENMP_FOUND)
           list(APPEND lulesh_par_deps openmp)
    endif()

    blt_add_executable(
        NAME        lulesh_par
        SOURCES     ${LULESH_SOURCES}
        DEPENDS_ON  ${lulesh_par_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    blt_add_target_compile_flags(TO lulesh_par FLAGS "-DUSE_MPI=1 ${lulesh_openmp_flags}")


    # install target for lulesh mpi
    install(TARGETS lulesh_par
            EXPORT  ascent
            LIBRARY DESTINATION examples/ascent/proxies/lulesh
            ARCHIVE DESTINATION examples/ascent/proxies/lulesh
            RUNTIME DESTINATION examples/ascent/proxies/lulesh
    )


endif()






