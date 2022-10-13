#==============================================================================
# See License.txt
#==============================================================================
set(vtkh_rendering_headers
    Annotator.hpp
    LineRenderer.hpp
    MeshRenderer.hpp
    RayTracer.hpp
    Render.hpp
    Renderer.hpp
    PointRenderer.hpp
    ScalarRenderer.hpp
    Scene.hpp
    VolumeRenderer.hpp
    )

set(vtkh_rendering_sources
    Annotator.cpp
    LineRenderer.cpp
    MeshRenderer.cpp
    RayTracer.cpp
    Render.cpp
    Renderer.cpp
    PointRenderer.cpp
    ScalarRenderer.cpp
    Scene.cpp
    VolumeRenderer.cpp
    )

###############################################################################
# always install headers
install(FILES ${vtkh_rendering_headers}
        DESTINATION include/vtkh/rendering)

###############################################################################
if (ENABLE_SERIAL)
    set(vtkh_rendering_deps vtkh_core vtkh_utils vtkh_filters vtkh_compositing ${vtkh_base_deps})

    blt_add_library(NAME vtkh_rendering
                    SOURCES ${vtkh_rendering_sources}
                    HEADERS ${vtkh_rendering_headers}
                    DEPENDS_ON ${vtkh_rendering_deps}
                    )

    vtkm_add_target_information(vtkh_rendering DEVICE_SOURCES ${vtkh_rendering_sources})

    # build time only defs
    target_compile_definitions(vtkh_rendering PRIVATE ASCENT_EXPORTS_FLAG)
    if(ENABLE_HIDDEN_VISIBILITY)
      set_target_properties(vtkh_rendering PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

    # Install libraries
    install(TARGETS vtkh_rendering
           EXPORT ascent
           ARCHIVE DESTINATION lib
           LIBRARY DESTINATION lib
           RUNTIME DESTINATION lib
           )

endif()

###############################################################################
if (MPI_FOUND)

    set(vtkh_rendering_mpi_deps vtkh_core_mpi vtkh_utils_mpi vtkh_filters_mpi vtkh_compositing_mpi ${vtkh_base_deps})

    blt_add_library(NAME vtkh_rendering_mpi
                    SOURCES ${vtkh_rendering_sources}
                    HEADERS ${vtkh_rendering_headers}
                    DEPENDS_ON ${vtkh_rendering_mpi_deps}
                    )

    vtkm_add_target_information(vtkh_rendering_mpi DEVICE_SOURCES ${vtkh_rendering_sources})

    # build time only defs
    target_compile_definitions(vtkh_rendering_mpi PRIVATE ASCENT_EXPORTS_FLAG)
#    set_target_properties(vtkh_rendering_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
    #
    target_compile_definitions(vtkh_rendering_mpi PRIVATE VTKH_PARALLEL)

    # Install libraries
    install(TARGETS vtkh_rendering_mpi
            EXPORT ascent
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION lib
            )

endif()
