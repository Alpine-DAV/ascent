set(subset_thirdparty_libs dray)
set(subset_thirdparty_libs_mpi dray_mpi)

################################################
# subset
################################################
if(ENABLE_SERIAL)
  blt_add_executable(
    NAME subset
    SOURCES subset.cpp
    DEPENDS_ON ${subset_thirdparty_libs}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()


################################################
# MPI SUPPORT
################################################

if(MPI_FOUND)
################################################
# subset mpi
################################################

  blt_add_executable(
    NAME subset_mpi
    SOURCES subset.cpp
    DEPENDS_ON ${subset_thirdparty_libs_mpi}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
  )

  target_compile_definitions(subset_mpi PRIVATE "MPI_ENABLED")
endif()

configure_file(subset_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/subset_config.yaml COPYONLY)
