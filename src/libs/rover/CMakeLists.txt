###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

# setup vars for config gen
if(CUDA_FOUND)
  set(ROVER_CUDA_ENABLED TRUE)
endif()

if(HIP_FOUND)
  set(ROVER_HIP_ENABLED TRUE)
endif()

if(ENABLE_OPENMP)
  set(ROVER_OPENMP_ENABLED TRUE)
endif()

if(KOKKOS_FOUND)
  set(ROVER_KOKKOS_ENABLED TRUE)
endif()

if(ENABLE_LOGGING)
  set(ROVER_ENABLE_LOGGING TRUE)
endif()

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/rover_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/rover_config.h")

############################################################
# setup base deps list
############################################################
set(rover_base_deps ascent_lodepng vtkm conduit::conduit)

if(CUDA_FOUND)
    # triggers cuda compile
    list(APPEND rover_base_deps cuda)
endif()

if(ENABLE_HIP)
    list(APPEND rover_base_deps blt_hip)
endif()

if(ENABLE_OPENMP)
    list(APPEND rover_base_deps ${ascent_blt_openmp_deps})
endif()

if(KOKKOS_FOUND)
    list(APPEND rover_base_deps Kokkos::kokkos)
endif()

set(rover_thirdparty_deps ${rover_base_deps}
                          vtkh)

set(rover_headers
      domain.hpp
      image.hpp
      partial_image.hpp
      rover_exports.h
      rover_exceptions.hpp
      rover_types.hpp
      rover.hpp
      scheduler.hpp
      scheduler_base.hpp
      static_scheduler.hpp
      # engines
      engine.hpp
      energy_engine.hpp
      volume_engine.hpp
      # ray generators headers
      ray_generators/ray_generator.hpp
      ray_generators/camera_generator.hpp
      ray_generators/visit_generator.hpp
      rover_exports.h
      vtkm_typedefs.hpp
      # utils headers
      utils/png_encoder.hpp
      utils/rover_logging.hpp
      utils/vtk_dataset_reader.hpp
      #
      ${CMAKE_CURRENT_BINARY_DIR}/rover_config.h
     )

set(rover_sources
      domain.cpp
      image.cpp
      rover.cpp
      scheduler.cpp
      scheduler_base.cpp
      # engines
      energy_engine.cpp
      volume_engine.cpp
      # ray generators
      ray_generators/ray_generator.cpp
      ray_generators/camera_generator.cpp
      ray_generators/visit_generator.cpp
      # utils sources
      utils/png_encoder.cpp
      utils/rover_logging.cpp
      utils/vtk_dataset_reader.cpp
 )

###############################################################################
if (ENABLE_SERIAL)
    blt_add_library(NAME rover
                    SOURCES ${rover_sources}
                    HEADERS ${rover_headers}
                    DEPENDS_ON ${rover_thirdparty_deps}
                    )

    vtkm_add_target_information(rover DEVICE_SOURCES ${rover_sources})

    # build includes
    # this allows us to include as <header.hpp>
    target_include_directories(rover PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/rover>)
    target_include_directories(rover PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/rover>)


    if(ENABLE_HIDDEN_VISIBILITY)
      set_target_properties(rover PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

    target_compile_definitions(rover PRIVATE ROVER_EXPORTS_FLAG)

    # install target for mpi ver
    install(TARGETS rover
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )

endif()

###################################################
#   Build parallel version
##################################################

if(MPI_FOUND)

  set(rover_mpi_thirdparty_deps ${rover_base_deps}
                                vtkh_mpi
                                conduit::conduit_mpi
                                ${ascent_blt_mpi_deps}
                                )

  blt_add_library(NAME rover_mpi
                  SOURCES ${rover_sources}
                  HEADERS ${rover_headers}
                  DEPENDS_ON ${rover_mpi_thirdparty_deps}
                  )

  vtkm_add_target_information(rover_mpi DEVICE_SOURCES ${rover_sources})

  # build includes
  # this allows us to include as <rover/header.hpp>
  target_include_directories(rover_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/rover>)
  target_include_directories(rover_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/rover>)

  if(ENABLE_HIDDEN_VISIBILITY)
    set_target_properties(rover_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
  endif()

  blt_add_target_compile_flags(TO rover_mpi FLAGS "-DROVER_PARALLEL")
  target_compile_definitions(rover_mpi PRIVATE ROVER_EXPORTS_FLAG)

  install(TARGETS rover_mpi
          EXPORT  ascent
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          RUNTIME DESTINATION lib
  )

endif()
