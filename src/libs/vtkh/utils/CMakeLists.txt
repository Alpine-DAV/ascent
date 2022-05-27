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

set(vtkh_utils_thirdparty_libs vtkm vtkh_lodepng)
if (ENABLE_SERIAL)
    if(CUDA_FOUND)
      list(APPEND vtkh_utils_thirdparty_libs cuda)
    endif()

    if(ENABLE_OPENMP)
        list(APPEND vtkh_utils_thirdparty_libs openmp)
    endif()

    if(ENABLE_KOKKOS)
        list(APPEND vtkh_utils_thirdparty_libs Kokkos::kokkos)
    endif()

    blt_add_library(
      NAME vtkh_utils
      SOURCES ${vtkh_utils_sources}
      HEADERS ${vtkh_utils_headers}
      DEPENDS_ON ${vtkh_utils_thirdparty_libs}
      )

    vtkm_add_target_information(vtkh_utils DEVICE_SOURCES vtkm_dataset_info.cpp)
    target_compile_definitions(vtkh_utils PRIVATE VTKH_COMPILING_FLAG)

    if(ENABLE_OPENMP)
        target_compile_definitions(vtkh_utils PRIVATE VTKH_USE_OPENMP)
    endif()

    # Install libraries
    install(TARGETS vtkh_utils
      EXPORT ${VTKh_EXPORT_NAME}
      ARCHIVE DESTINATION ${VTKh_INSTALL_LIB_DIR}
      LIBRARY DESTINATION ${VTKh_INSTALL_LIB_DIR}
      RUNTIME DESTINATION ${VTKh_INSTALL_BIN_DIR}
      )

    # Install headers
    install(FILES ${vtkh_utils_headers}
            DESTINATION ${VTKh_INSTALL_INCLUDE_DIR}/vtkh/utils)
endif()

if (MPI_FOUND)

    blt_add_library(
      NAME vtkh_utils_mpi
      SOURCES ${vtkh_utils_sources}
      HEADERS ${vtkh_utils_headers}
      DEPENDS_ON ${vtkh_utils_thirdparty_libs} mpi
      )

    target_compile_definitions(vtkh_utils_mpi PRIVATE VTKH_PARALLEL)
    target_compile_definitions(vtkh_utils_mpi PRIVATE VTKH_COMPILING_FLAG)

    vtkm_add_target_information(vtkh_utils_mpi DEVICE_SOURCES vtkm_dataset_info.cpp)

    if(ENABLE_OPENMP)
        target_compile_definitions(vtkh_utils_mpi PRIVATE VTKH_USE_OPENMP)
    endif()

    # Install libraries
    install(TARGETS vtkh_utils_mpi
      EXPORT ${VTKh_EXPORT_NAME}
      ARCHIVE DESTINATION ${VTKh_INSTALL_LIB_DIR}
      LIBRARY DESTINATION ${VTKh_INSTALL_LIB_DIR}
      RUNTIME DESTINATION ${VTKh_INSTALL_BIN_DIR}
      )

    if(NOT ENABLE_SERIAL)
      # Install headers
      install(FILES ${vtkh_utils_headers}
              DESTINATION ${VTKh_INSTALL_INCLUDE_DIR}/vtkh/utils)
    endif()
endif()
