#==============================================================================
# See License.txt
#==============================================================================
set(vtkh_compositing_headers
    Image.hpp
    ImageCompositor.hpp
    Compositor.hpp
    PartialCompositor.hpp
    PayloadCompositor.hpp
    PayloadImage.hpp
    AbsorptionPartial.hpp
    EmissionPartial.hpp
    VolumePartial.hpp
    )

set(vtkh_compositing_sources
    Image.cpp
    Compositor.cpp
    PartialCompositor.cpp
    PayloadCompositor.cpp
    )

# always install headers
install(FILES ${vtkh_compositing_headers}
        DESTINATION include/vtkh/compositing)

###############################################################################
if (ENABLE_SERIAL)
    set(vtkh_compositing_deps vtkh_core vtkh_utils)

    if(ENABLE_OPENMP)
        list(APPEND vtkh_compositing_deps ${ascent_blt_openmp_deps})
    endif()

    blt_add_library(NAME vtkh_compositing
                    SOURCES ${vtkh_compositing_sources}
                    HEADERS ${vtkh_compositing_headers}
                    DEPENDS_ON ${vtkh_compositing_deps}
                    )

    # build time only defs
    target_compile_definitions(vtkh_compositing PRIVATE ASCENT_EXPORTS_FLAG)
    if(ENABLE_HIDDEN_VISIBILITY)
      set_target_properties(vtkh_compositing PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

    # build includes
    # this allows us to include as <vtkh/header.hpp>
    target_include_directories(vtkh_compositing PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(vtkh_compositing PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(vtkh_compositing PUBLIC $<INSTALL_INTERFACE:include/ascent/vtkh>)

    # Install libraries
    install(TARGETS vtkh_compositing
            EXPORT ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
            )

endif()


###############################################################################
if (MPI_FOUND)

    set(vtkh_compositing_mpi_headers
        DirectSendCompositor.hpp
        MPICollect.hpp
        RadixKCompositor.hpp
        vtkh_diy_collect.hpp
        vtkh_diy_image_block.hpp
        vtkh_diy_utils.hpp
        PartialCompositor.hpp
        PayloadCompositor.hpp
        PayloadImage.hpp
        )

    set(vtkh_compositing_mpi_sources
        DirectSendCompositor.cpp
        RadixKCompositor.cpp
        PartialCompositor.cpp
        PayloadCompositor.cpp
        )

    set(vtkh_compositing_mpi_deps vtkh_core_mpi vtkh_utils_mpi)

    if(ENABLE_OPENMP)
        list(APPEND vtkh_compositing_mpi_deps ${ascent_blt_openmp_deps})
    endif()

    blt_add_library(NAME vtkh_compositing_mpi
                    SOURCES ${vtkh_compositing_sources} ${vtkh_compositing_mpi_sources}
                    HEADERS ${vtkh_compositing_headers} ${vtkh_compositing_mpi_headers}
                    DEPENDS_ON ${vtkh_compositing_mpi_deps} ${ascent_blt_mpi_deps}
                    )

    # build time only defs
    target_compile_definitions(vtkh_compositing_mpi PRIVATE ASCENT_EXPORTS_FLAG)
#    set_target_properties(vtkh_compositing_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
    #
    target_compile_definitions(vtkh_compositing_mpi PRIVATE VTKH_PARALLEL)

    # build includes
    # this allows us to include as <vtkh/header.hpp>
    target_include_directories(vtkh_compositing_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(vtkh_compositing_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(vtkh_compositing_mpi PUBLIC $<INSTALL_INTERFACE:include/ascent/vtkh>)

    # internal diy include
    target_include_directories(vtkh_compositing_mpi PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/diy/include/>)

    # Install libraries
    install(TARGETS vtkh_compositing_mpi
            EXPORT ascent
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION lib)

    # Install extra mpi related headers
    install(FILES ${vtkh_compositing_mpi_headers}
            DESTINATION include/vtkh/compositing)

endif()
