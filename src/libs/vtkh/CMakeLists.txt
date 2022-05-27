#==============================================================================
# See License.txt
#==============================================================================
####################################
# Hide symbols for all vtkh targets
####################################

add_subdirectory(thirdparty_builtin)

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/vtkh_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/vtkh_config.h")


add_subdirectory(utils)

set(vtkh_core_headers
  DataSet.hpp
  Error.hpp
  Logger.hpp
  Timer.hpp
  StatisticsDB.hpp
  vtkh.hpp
  vtkh_exports.h
  ${CMAKE_CURRENT_BINARY_DIR}/vtkh_config.h
  )

set(vtkh_core_sources
  DataSet.cpp
  Logger.cpp
  Timer.cpp
  StatisticsDB.cpp
  vtkh.cpp
  )


set(vtkh_core_deps vtkm vtkh_utils)


if(CUDA_FOUND)
  list(APPEND vtkh_core_deps cuda)
endif()

if(ENABLE_OPENMP)
    list(APPEND vtkm_core_deps ${ascent_blt_openmp_deps})
endif()

if(KOKKOS_FOUND)
  list(APPEND vtkh_core_deps Kokkos::kokkos)
endif()

if (ENABLE_SERIAL)
    # blt vtk dep, includes all of vtkm
    blt_add_library(
      NAME vtkh_core
      SOURCES ${vtkh_core_sources}
      HEADERS ${vtkh_core_headers}
      DEPENDS_ON ${vtkh_core_deps}
      )

    # build time only defs
    vtkm_add_target_information(vtkh_core DEVICE_SOURCES ${vtkh_core_sources})
    if(ENABLE_LOGGING)
      target_compile_definitions(vtkh_core PRIVATE VTKH_ENABLE_LOGGING)
    endif()
  
    target_compile_definitions(vtkh_core PRIVATE ASCENT_EXPORTS_FLAG)
    set_target_properties(vtkh_core PROPERTIES CXX_VISIBILITY_PRESET hidden)

    # build includes
    # this allows us to include as <vtkh/header.hpp>
    target_include_directories(vtkh_core PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(vtkh_core PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(vtkh_core PUBLIC $<INSTALL_INTERFACE:include/ascent/vtkh>)
  
  
    # Install libraries
    install(TARGETS vtkh_core
            EXPORT ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
            )
endif()

# Install headers
install(FILES ${vtkh_core_headers}
  DESTINATION ${VTKh_INSTALL_INCLUDE_DIR}/vtkh)

if(MPI_FOUND)

  set(vtkh_core_mpi_deps vtkm vtkh_utils_mpi)

  if(CUDA_FOUND)
    list(APPEND vtkh_core_mpi_deps cuda)
  endif()

  if(ENABLE_OPENMP)
      list(APPEND vtkm_core_deps ${ascent_blt_openmp_deps})
  endif()

  if(KOKKOS_FOUND)
    list(APPEND vtkh_core_mpi_deps Kokkos::kokkos)
  endif()


  blt_add_library(
      NAME vtkh_core_mpi
      SOURCES ${vtkh_core_sources}
      HEADERS ${vtkh_core_headers}
      DEPENDS_ON ${vtkh_core_mpi_deps} mpi
      )

  vtkm_add_target_information(vtkh_core_mpi DEVICE_SOURCES ${vtkh_core_sources})


  target_compile_definitions(vtkh_core_mpi PRIVATE ASCENT_EXPORTS_FLAG)
  set_target_properties(vtkh_core_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
  target_compile_definitions(vtkh_core_mpi PRIVATE VTKH_PARALLEL)

  # build includes
  # this allows us to include as <vtkh/header.hpp>
  target_include_directories(vtkh_core_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
  target_include_directories(vtkh_core_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

  # install includes
  # setup exported target to point to proper include dir
  target_include_directories(vtkh_core_mpi PUBLIC $<INSTALL_INTERFACE:include/ascent/vtkh>)


  # Install libraries
  install(TARGETS vtkh_core_mpi
          EXPORT ascent
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          RUNTIME DESTINATION lib
          )
endif()

add_subdirectory(vtkm_filters)
add_subdirectory(filters)
add_subdirectory(rendering)
add_subdirectory(compositing)
