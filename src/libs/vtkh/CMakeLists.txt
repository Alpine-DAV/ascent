#==============================================================================
# See License.txt
#==============================================================================

# setup vars for config gen
if(CUDA_FOUND)
  set(VTKH_CUDA_ENABLED TRUE)
endif()

if(HIP_FOUND)
  set(VTKH_HIP_ENABLED TRUE)
endif()

if(ENABLE_OPENMP)
  set(VTKH_OPENMP_ENABLED TRUE)
endif()

if(KOKKOS_FOUND)
  set(VTKH_KOKKOS_ENABLED TRUE)
endif()

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/vtkh_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/vtkh_config.h")

############################################################
# setup base deps list
############################################################
set(vtkh_base_deps )

if(CUDA_FOUND)
    # triggers cuda compile
    list(APPEND vtkh_base_deps cuda)
endif()

if(ENABLE_HIP)
    list(APPEND vtkh_base_deps blt_hip)
endif()

if(ENABLE_OPENMP)
    list(APPEND vtkh_base_deps ${ascent_blt_openmp_deps})
endif()

if(KOKKOS_FOUND)
    list(APPEND vtkh_base_deps Kokkos::kokkos)
endif()



add_subdirectory(utils)


###############################################################################
# core lib
###############################################################################
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


# always install headers
install(FILES ${vtkh_core_headers}
        DESTINATION include/ascent/vtkh)

###############################################################################
if (ENABLE_SERIAL)

    set(vtkh_core_deps vtkm vtkh_utils ${vtkh_base_deps})

    blt_add_library(NAME vtkh_core
                    SOURCES ${vtkh_core_sources}
                    HEADERS ${vtkh_core_headers}
                    DEPENDS_ON ${vtkh_core_deps}
                    )


    vtkm_add_target_information(vtkh_core DEVICE_SOURCES ${vtkh_core_sources})

    # build time only defs
    target_compile_definitions(vtkh_core PRIVATE ASCENT_EXPORTS_FLAG)
    if(ENABLE_HIDDEN_VISIBILITY)
      set_target_properties(vtkh_core PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

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


###############################################################################
if(MPI_FOUND)

    set(vtkh_core_mpi_deps vtkm vtkh_utils_mpi ${vtkh_base_deps})

    blt_add_library(NAME vtkh_core_mpi
                    SOURCES ${vtkh_core_sources}
                    HEADERS ${vtkh_core_headers}
                    DEPENDS_ON ${vtkh_core_mpi_deps} ${ascent_blt_mpi_deps}
                    )

    vtkm_add_target_information(vtkh_core_mpi DEVICE_SOURCES ${vtkh_core_sources})

    # build time only defs
    target_compile_definitions(vtkh_core_mpi PRIVATE ASCENT_EXPORTS_FLAG)
    if(ENABLE_HIDDEN_VISIBILITY)
      set_target_properties(vtkh_core_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()
    #
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


#------------------------------------------------------------------------------
# Add interface targets that makes it easier to depend on VTKh
if (ENABLE_SERIAL)
  add_library(vtkh INTERFACE)
  target_link_libraries(vtkh INTERFACE
                        vtkh_core
                        vtkh_utils
                        vtkh_filters
                        vtkh_rendering)

  # Install libraries
  install(TARGETS vtkh EXPORT ascent)
endif()

if (MPI_FOUND)
  add_library(vtkh_mpi INTERFACE)
  target_link_libraries(vtkh_mpi INTERFACE
                        vtkh_core_mpi
                        vtkh_utils_mpi
                        vtkh_filters_mpi
                        vtkh_rendering_mpi)

  # Install libraries
  install(TARGETS vtkh_mpi EXPORT ascent)
endif()
