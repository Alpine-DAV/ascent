# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

set(furnace_thirdparty_libs dray)
set(furnace_thirdparty_libs_mpi dray_mpi)


################################################
# openmp support
################################################
if(ENABLE_OPENMP)
    list(APPEND furnace_thirdparty_libs ${ascent_blt_openmp_deps})
    list(APPEND furnace_thirdparty_libs_mpi ${ascent_blt_openmp_deps})
endif()

if(ENABLE_SERIAL)
    ################################################
    # point location furnace
    ################################################
    blt_add_executable(NAME point_location
                       SOURCES point_location.cpp
                       DEPENDS_ON ${furnace_thirdparty_libs}
                       OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                       )

    ################################################
    # scalar renderer furnace
    ################################################
    blt_add_executable(NAME scalar_renderer
                       SOURCES scalar_renderer.cpp
                       DEPENDS_ON ${furnace_thirdparty_libs}
                       OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                       )

    configure_file(point_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/point_config.yaml COPYONLY)

    ################################################
    # surface intersection furnace
    ################################################
    blt_add_executable(NAME surface_intersection
                       SOURCES surface_intersection.cpp
                       DEPENDS_ON ${furnace_thirdparty_libs}
                       OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                       )

    configure_file(intersection_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/intersection_config.yaml COPYONLY)

    ################################################
    # volume intersection furnace
    ################################################
    blt_add_executable(NAME volume_rendering
                      SOURCES volume_rendering.cpp
                      DEPENDS_ON ${furnace_thirdparty_libs}
                      OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                      )

    #configure_file(point_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/point_config.yaml COPYONLY)

    # TODO:
    # install(FILES point_config.yaml intersection_config.yaml
    # DESTINATION utilities/furnace
    # )

################################################
# end serial support
################################################
endif()

################################################
# MPI SUPPORT
################################################

if(MPI_FOUND)
    ################################################
    # point location furnace
    ################################################
    blt_add_executable(NAME point_location_mpi
                       SOURCES point_location.cpp
                       DEPENDS_ON ${furnace_thirdparty_libs_mpi}
                       OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                       )

    target_compile_definitions(point_location_mpi PRIVATE "MPI_ENABLED")

    ################################################
    # scalar renderer furnace
    ################################################
    blt_add_executable(NAME scalar_renderer_mpi
                       SOURCES scalar_renderer.cpp
                       DEPENDS_ON ${furnace_thirdparty_libs_mpi}
                       OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                       )

    target_compile_definitions(scalar_renderer_mpi PRIVATE "MPI_ENABLED")

    ################################################
    # surface intersection furnace
    ################################################
    blt_add_executable(NAME surface_intersection_mpi
                       SOURCES surface_intersection.cpp
                       DEPENDS_ON ${furnace_thirdparty_libs_mpi}
                       OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                       )

    target_compile_definitions(surface_intersection_mpi PRIVATE "MPI_ENABLED")

    ################################################
    # volume intersection furnace
    ################################################
    blt_add_executable(NAME volume_rendering_mpi
                       SOURCES volume_rendering.cpp
                       DEPENDS_ON ${furnace_thirdparty_libs_mpi}
                       OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                       )

    target_compile_definitions(volume_rendering_mpi PRIVATE "MPI_ENABLED")

endif()
