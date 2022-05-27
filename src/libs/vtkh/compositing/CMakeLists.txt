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

if (ENABLE_SERIAL)
    set(vtkh_compositing_deps vtkh_core vtkh_utils)

    if(ENABLE_OPENMP)
        list(APPEND vtkh_compositing_deps openmp)
    endif()

    blt_add_library(
      NAME vtkh_compositing
      SOURCES ${vtkh_compositing_sources}
      HEADERS ${vtkh_compositing_headers}
      DEPENDS_ON ${vtkh_compositing_deps}
      )

    target_compile_definitions(vtkh_compositing PRIVATE VTKH_COMPILING_FLAG)

    if(ENABLE_OPENMP)
        target_compile_definitions(vtkh_compositing PRIVATE VTKH_USE_OPENMP)
    endif()

    if(ENABLE_LOGGING)
      target_compile_definitions(vtkh_compositing PRIVATE VTKH_ENABLE_LOGGING)
    endif()

    # Install libraries
    install(TARGETS vtkh_compositing
      EXPORT ${VTKh_EXPORT_NAME}
      ARCHIVE DESTINATION ${VTKh_INSTALL_LIB_DIR}
      LIBRARY DESTINATION ${VTKh_INSTALL_LIB_DIR}
      RUNTIME DESTINATION ${VTKh_INSTALL_BIN_DIR}
      )

    # Install headers
    install(FILES ${vtkh_compositing_headers}
      DESTINATION ${VTKh_INSTALL_INCLUDE_DIR}/vtkh/compositing)
endif()

#------------------------------------------------------------------------------
# Handle parallel library
#------------------------------------------------------------------------------
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

if (MPI_FOUND)

    set(vtkh_compositing_mpi_deps vtkh_core_mpi vtkh_utils_mpi)

    if(ENABLE_OPENMP)
         list(APPEND vtkh_compositing_mpi_deps openmp)
    endif()


    blt_add_library(
        NAME vtkh_compositing_mpi
        SOURCES ${vtkh_compositing_sources} ${vtkh_compositing_mpi_sources}
        HEADERS ${vtkh_compositing_headers} ${vtkh_compositing_mpi_headers}
        DEPENDS_ON ${vtkh_compositing_mpi_deps}
    )

    #target_link_libraries(vtkh_rendering_mpi PRIVATE vtkhdiy)
    #this is a workaround for a cmake issue linking an interface header only library statically
    target_include_directories(vtkh_compositing_mpi PRIVATE $<TARGET_PROPERTY:vtkhdiy,INTERFACE_INCLUDE_DIRECTORIES>)

    target_compile_definitions(vtkh_compositing_mpi PRIVATE VTKH_COMPILING_FLAG)
    target_compile_definitions(vtkh_compositing_mpi PRIVATE VTKH_PARALLEL)

    if(ENABLE_OPENMP)
      target_compile_definitions(vtkh_compositing_mpi PRIVATE VTKH_USE_OPENMP)
    endif()

    if(ENABLE_LOGGING)
      target_compile_definitions(vtkh_compositing_mpi PRIVATE VTKH_ENABLE_LOGGING)
    endif()


    vtkm_add_target_information(vtkh_compositing_mpi DEVICE_SOURCES ${vtkh_rendering_sources})

    # Install libraries
    install(TARGETS vtkh_compositing_mpi
            EXPORT ${VTKh_EXPORT_NAME}
            ARCHIVE DESTINATION ${VTKh_INSTALL_LIB_DIR}
            LIBRARY DESTINATION ${VTKh_INSTALL_LIB_DIR}
            RUNTIME DESTINATION ${VTKh_INSTALL_BIN_DIR})

    set(vtkh_compositing_mpi_install_headers ${vtkh_compositing_mpi_headers})
    if(NOT ENABLE_SERIAL)
      list(APPEND vtkh_compositing_mpi_install_headers ${vtkh_compositing_headers})
    endif()
    # Install headers
    install(FILES ${vtkh_compositing_mpi_install_headers}
            DESTINATION ${VTKh_INSTALL_INCLUDE_DIR}/vtkh/compositing)

endif()
