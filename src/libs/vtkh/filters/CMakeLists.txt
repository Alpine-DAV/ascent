#==============================================================================
# See License.txt
#==============================================================================

set(vtkh_filters_headers
  Filter.hpp
  CellAverage.hpp
  CleanGrid.hpp
  Clip.hpp
  ClipField.hpp
  CompositeVector.hpp
  Gradient.hpp
  GhostStripper.hpp
  HistSampling.hpp
  Histogram.hpp
  Log.hpp
  IsoVolume.hpp
  NoOp.hpp
  Lagrangian.hpp
  MarchingCubes.hpp
  MeshQuality.hpp
  ParticleAdvection.hpp
  ParticleMerging.hpp
  PointAverage.hpp
  PointTransform.hpp
  Recenter.hpp
  Tetrahedralize.hpp
  Threshold.hpp
  Triangulate.hpp
  Statistics.hpp
  Streamline.hpp
  Slice.hpp
  VectorComponent.hpp
  VectorMagnitude.hpp
  )

if(ENABLE_FILTER_CONTOUR_TREE)
    list(APPEND vtkh_filters_headers ContourTree.hpp)
endif()

set(vtkh_filters_sources
  Filter.cpp
  CellAverage.cpp
  CleanGrid.cpp
  Clip.cpp
  ClipField.cpp
  CompositeVector.cpp
  Gradient.cpp
  GhostStripper.cpp
  HistSampling.cpp
  Histogram.cpp
  Log.cpp
  IsoVolume.cpp
  NoOp.cpp
  Lagrangian.cpp
  MarchingCubes.cpp
  MeshQuality.cpp
  ParticleAdvection.cpp
  ParticleMerging.cpp
  PointAverage.cpp
  PointTransform.cpp
  Recenter.cpp
  Tetrahedralize.cpp
  Threshold.cpp
  Triangulate.cpp
  Slice.cpp
  Statistics.cpp
  Streamline.cpp
  VectorComponent.cpp
  VectorMagnitude.cpp
  )

if(ENABLE_FILTER_CONTOUR_TREE)
  list(APPEND vtkh_filters_sources ContourTree.cpp)
endif()

if (ENABLE_SERIAL)
    set(vtkh_filters_deps vtkh_core vtkh_utils vtkm_compiled_filters)

    if(CUDA_FOUND)
        # triggers cuda compile
        list(APPEND vtkh_filters_deps cuda)
    endif()

    if(ENABLE_OPENMP)
        list(APPEND vtkh_filters_deps openmp)
    endif()

    if(ENABLE_KOKKOS)
        list(APPEND vtkh_filters_deps Kokkos::kokkos)
    endif()

    blt_add_library(
      NAME vtkh_filters
      SOURCES ${vtkh_filters_sources}
      HEADERS ${vtkh_filters_headers}
      DEPENDS_ON ${vtkh_filters_deps}
      )

    target_compile_definitions(vtkh_filters PRIVATE VTKH_COMPILING_FLAG)

    if(ENABLE_FILTER_CONTOUR_TREE)
      target_compile_definitions(vtkh_filters PRIVATE VTK_H_ENABLE_FILTER_CONTOUR_TREE)
    endif()

    if(ENABLE_OPENMP)
        target_compile_definitions(vtkh_filters PRIVATE VTKH_USE_OPENMP)
    endif()

    if(ENABLE_LOGGING)
      target_compile_definitions(vtkh_filters PRIVATE VTKH_ENABLE_LOGGING)
    endif()

    if(ENABLE_CUDA)
        set_target_properties(vtkh_filters PROPERTIES LINKER_LANGUAGE CUDA)
    endif()

    vtkm_add_target_information(vtkh_filters DEVICE_SOURCES ${vtkh_filter_sources})

    # Install libraries
    install(TARGETS vtkh_filters
      EXPORT ${VTKh_EXPORT_NAME}
      ARCHIVE DESTINATION ${VTKh_INSTALL_LIB_DIR}
      LIBRARY DESTINATION ${VTKh_INSTALL_LIB_DIR}
      RUNTIME DESTINATION ${VTKh_INSTALL_BIN_DIR}
      )


    # Install headers
    install(FILES ${vtkh_filters_headers}
      DESTINATION ${VTKh_INSTALL_INCLUDE_DIR}/vtkh/filters)
endif()

if (MPI_FOUND)

  set(vtkh_filters_mpi_deps vtkh_core_mpi vtkh_utils_mpi vtkm_compiled_filters)

    # necessary b/c of how find_cuda works
    if(CUDA_FOUND)
         # triggers cuda compile
         list(APPEND vtkh_filters_mpi_deps cuda)
    endif()

    if(ENABLE_OPENMP)
         list(APPEND vtkh_filters_mpi_deps openmp)
    endif()

    if(ENABLE_KOKKOS)
        list(APPEND vtkh_filters_mpi_deps Kokkos::kokkos)
    endif()

    blt_add_library(
      NAME vtkh_filters_mpi
      SOURCES ${vtkh_filters_sources} ${vtkh_mpi_sources}
      HEADERS ${vtkh_filters_headers} ${vtkh_mpi_headers}
      DEPENDS_ON ${vtkh_filters_mpi_deps}
      )

    target_compile_definitions(vtkh_filters_mpi PRIVATE VTKH_PARALLEL)
    target_compile_definitions(vtkh_filters_mpi PRIVATE VTKH_COMPILING_FLAG)

    if(ENABLE_FILTER_CONTOUR_TREE)
      target_compile_definitions(vtkh_filters_mpi PRIVATE VTK_H_ENABLE_FILTER_CONTOUR_TREE)
    endif()

    if(ENABLE_OPENMP)
          target_compile_definitions(vtkh_filters_mpi PRIVATE VTKH_USE_OPENMP)
    endif()

    if(ENABLE_LOGGING)
      target_compile_definitions(vtkh_filters_mpi PRIVATE VTKH_ENABLE_LOGGING)
    endif()

    vtkm_add_target_information(vtkh_filters_mpi DEVICE_SOURCES ${vtkh_filter_sources})

    if(ENABLE_CUDA)
        set_target_properties(vtkh_filters_mpi PROPERTIES LINKER_LANGUAGE CUDA)
    endif()


    # Install libraries
    install(TARGETS vtkh_filters_mpi
            EXPORT ${VTKh_EXPORT_NAME}
            ARCHIVE DESTINATION ${VTKh_INSTALL_LIB_DIR}
            LIBRARY DESTINATION ${VTKh_INSTALL_LIB_DIR}
            RUNTIME DESTINATION ${VTKh_INSTALL_BIN_DIR})

    if(NOT ENABLE_SERIAL)
      # Install headers
      install(FILES ${vtkh_filters_headers}
        DESTINATION ${VTKh_INSTALL_INCLUDE_DIR}/vtkh/filters)
    endif()
endif()
