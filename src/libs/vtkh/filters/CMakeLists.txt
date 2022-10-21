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

if(VTKH_ENABLE_FILTER_CONTOUR_TREE)
    list(APPEND vtkh_filters_sources ContourTree.cpp)
endif()

############################################################
# setup base deps list, shared between mpi and non mpi lib
############################################################
set(vtkh_filters_base_deps vtkm_compiled_filters ${vtkh_base_deps})

############################################################
# Always install headers
############################################################
install(FILES ${vtkh_filters_headers}
        DESTINATION include/vtkh/filters)

############################################################
if (ENABLE_SERIAL)
    set(vtkh_filters_deps vtkh_core vtkh_utils ${vtkh_filters_base_deps})


    blt_add_library(NAME vtkh_filters
                    SOURCES ${vtkh_filters_sources}
                    HEADERS ${vtkh_filters_headers}
                    DEPENDS_ON ${vtkh_filters_deps}
                    )

    vtkm_add_target_information(vtkh_filters DEVICE_SOURCES ${vtkh_filter_sources})

    # build time only defs
    target_compile_definitions(vtkh_filters PRIVATE ASCENT_EXPORTS_FLAG)
    if(ENABLE_HIDDEN_VISIBILITY)
      set_target_properties(vtkh_filters PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

    if(CUDA_FOUND)
        set_target_properties(vtkh_filters PROPERTIES LINKER_LANGUAGE CUDA)
    endif()

    # build includes
    # this allows us to include as <vtkh/header.hpp>
    target_include_directories(vtkh_filters PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(vtkh_filters PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(vtkh_filters PUBLIC $<INSTALL_INTERFACE:include/ascent/vtkh>)
  
    # Install libraries
    install(TARGETS vtkh_filters
            EXPORT ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
            )

endif()

############################################################
if (MPI_FOUND)

  set(vtkh_filters_mpi_deps vtkh_core_mpi vtkh_utils_mpi ${vtkh_filters_base_deps})

    blt_add_library(NAME vtkh_filters_mpi
                    SOURCES ${vtkh_filters_sources} ${vtkh_mpi_sources}
                    HEADERS ${vtkh_filters_headers} ${vtkh_mpi_headers}
                    DEPENDS_ON ${vtkh_filters_mpi_deps} ${ascent_blt_mpi_deps}
                    )
    
    # if serial is off, still still need to set this
    vtkm_add_target_information(vtkh_filters_mpi DEVICE_SOURCES ${vtkh_filter_sources})

    # build time only defs
    target_compile_definitions(vtkh_filters_mpi PRIVATE ASCENT_EXPORTS_FLAG)
    if(ENABLE_HIDDEN_VISIBILITY)
      set_target_properties(vtkh_filters_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()
    #
    target_compile_definitions(vtkh_filters_mpi PRIVATE VTKH_PARALLEL)

    if(CUDA_FOUND)
       set_target_properties(vtkh_filters_mpi PROPERTIES LINKER_LANGUAGE CUDA)
    endif()

    # build includes
    # this allows us to include as <vtkh/header.hpp>
    target_include_directories(vtkh_filters_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(vtkh_filters_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(vtkh_filters_mpi PUBLIC $<INSTALL_INTERFACE:include/ascent/vtkh>)

    # Install libraries
    install(TARGETS vtkh_filters_mpi
            EXPORT ascent
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION lib)

endif()
