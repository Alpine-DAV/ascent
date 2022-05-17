set(furnace_thirdparty_libs dray)
set(furnace_thirdparty_libs_mpi dray_mpi)

################################################
# openmp support
################################################
if(ENABLE_OPENMP)
  list(APPEND furnace_thirdparty_libs openmp)
  list(APPEND furnace_thirdparty_libs_mpi openmp)
endif()

if(ENABLE_SERIAL)
################################################
# point location furnace
################################################
  blt_add_executable(
    NAME point_location
    SOURCES point_location.cpp
    DEPENDS_ON ${furnace_thirdparty_libs}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )

  if(ENABLE_STATS)
    target_compile_definitions(point_location PRIVATE "DRAY_STATS")
  endif()

################################################
# scalar renderer furnace
################################################
  blt_add_executable(
    NAME scalar_renderer
    SOURCES scalar_renderer.cpp
    DEPENDS_ON ${furnace_thirdparty_libs}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )

  if(ENABLE_STATS)
    target_compile_definitions(scalar_renderer PRIVATE "DRAY_STATS")
  endif()

  configure_file(point_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/point_config.yaml COPYONLY)

################################################
# surface intersection furnace
################################################
  blt_add_executable(
    NAME surface_intersection
    SOURCES surface_intersection.cpp
    DEPENDS_ON ${furnace_thirdparty_libs}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )

  if(ENABLE_STATS)
    target_compile_definitions(surface_intersection PRIVATE "DRAY_STATS")
  endif()

  configure_file(intersection_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/intersection_config.yaml COPYONLY)

################################################
# volume intersection furnace
################################################
  blt_add_executable(
    NAME volume_rendering
    SOURCES volume_rendering.cpp
    DEPENDS_ON ${furnace_thirdparty_libs}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )

  if(ENABLE_STATS)
    target_compile_definitions(volume_rendering PRIVATE "DRAY_STATS")
  endif()

#configure_file(point_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/point_config.yaml COPYONLY)

  install(FILES point_config.yaml intersection_config.yaml
          DESTINATION utilities/furnace
          )

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
  blt_add_executable(
    NAME point_location_mpi
    SOURCES point_location.cpp
    DEPENDS_ON ${furnace_thirdparty_libs_mpi}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )

  target_compile_definitions(point_location_mpi PRIVATE "MPI_ENABLED")

  if(ENABLE_STATS)
    target_compile_definitions(point_location_mpi PRIVATE "DRAY_STATS")
  endif()

################################################
# scalar renderer furnace
################################################
  blt_add_executable(
    NAME scalar_renderer_mpi
    SOURCES scalar_renderer.cpp
    DEPENDS_ON ${furnace_thirdparty_libs_mpi}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )

  target_compile_definitions(scalar_renderer_mpi PRIVATE "MPI_ENABLED")

  if(ENABLE_STATS)
    target_compile_definitions(scalar_renderer_mpi PRIVATE "DRAY_STATS")
  endif()

################################################
# surface intersection furnace
################################################
  blt_add_executable(
    NAME surface_intersection_mpi
    SOURCES surface_intersection.cpp
    DEPENDS_ON ${furnace_thirdparty_libs_mpi}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )

  target_compile_definitions(surface_intersection_mpi PRIVATE "MPI_ENABLED")

  if(ENABLE_STATS)
    target_compile_definitions(surface_intersection_mpi PRIVATE "DRAY_STATS")
  endif()

################################################
# volume intersection furnace
################################################
  blt_add_executable(
    NAME volume_rendering_mpi
    SOURCES volume_rendering.cpp
    DEPENDS_ON ${furnace_thirdparty_libs_mpi}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )

  target_compile_definitions(volume_rendering_mpi PRIVATE "MPI_ENABLED")

  if(ENABLE_STATS)
    target_compile_definitions(volume_rendering_mpi PRIVATE "DRAY_STATS")
  endif()

endif()
