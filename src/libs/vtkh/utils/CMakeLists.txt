#==============================================================================
# See License.txt
#==============================================================================
set(vtkh_utils_headers
    Mutex.hpp
    PNGEncoder.hpp
    StreamUtil.hpp
    ThreadSafeContainer.hpp
    vtkm_array_utils.hpp
    vtkm_dataset_info.hpp
    )

set(vtkh_utils_sources
    PNGEncoder.cpp
    Mutex.cpp
    vtkm_dataset_info.cpp
    )


###############################################################################
 # always install headers
 install(FILES ${vtkh_utils_headers}
         DESTINATION include/vtkh/utils)

# deps are same between mpi and non mpi
set(vtkh_utils_deps vtkm ascent_lodepng ${vtkh_base_deps})

###############################################################################
if (ENABLE_SERIAL)
    blt_add_library(NAME vtkh_utils
                    SOURCES ${vtkh_utils_sources}
                    HEADERS ${vtkh_utils_headers}
                    DEPENDS_ON ${vtkh_utils_deps}
                    )

    vtkm_add_target_information(vtkh_utils DEVICE_SOURCES vtkm_dataset_info.cpp)

    # build time only defs
    target_compile_definitions(vtkh_utils PRIVATE ASCENT_EXPORTS_FLAG)
    set_target_properties(vtkh_utils PROPERTIES CXX_VISIBILITY_PRESET hidden)

    if(CUDA_FOUND)
        set_target_properties(vtkh_utils PROPERTIES LINKER_LANGUAGE CUDA)
    endif()

    # build includes
    # this allows us to include as <vtkh/header.hpp>
    target_include_directories(vtkh_utils PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(vtkh_utils PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(vtkh_utils PUBLIC $<INSTALL_INTERFACE:include/ascent/vtkh>)

    # Install libraries
    install(TARGETS vtkh_utils
            EXPORT ascent
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION lib
            )

endif()

###############################################################################
if (MPI_FOUND)

    blt_add_library(NAME vtkh_utils_mpi
                    SOURCES ${vtkh_utils_sources}
                    HEADERS ${vtkh_utils_headers}
                    DEPENDS_ON ${vtkh_utils_deps} ${ascent_blt_mpi_deps}
                    )

    vtkm_add_target_information(vtkh_utils_mpi DEVICE_SOURCES vtkm_dataset_info.cpp)

    # build time only defs
    target_compile_definitions(vtkh_utils_mpi PRIVATE ASCENT_EXPORTS_FLAG)
    set_target_properties(vtkh_utils_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
    #
    target_compile_definitions(vtkh_utils_mpi PRIVATE VTKH_PARALLEL)

    # build includes
    # this allows us to include as <vtkh/header.hpp>
    target_include_directories(vtkh_utils_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(vtkh_utils_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(vtkh_utils_mpi PUBLIC $<INSTALL_INTERFACE:include/ascent/vtkh>)

    # Install libraries
    install(TARGETS vtkh_utils_mpi
            EXPORT ascent
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION lib
            )

endif()
