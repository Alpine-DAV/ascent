# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

set(apcomp_headers
  apcomp.hpp
  apcomp_exports.h
  bounds.hpp
  error.hpp
  image.hpp
  compositor.hpp
  partial_compositor.hpp
  scalar_compositor.hpp
  scalar_image.hpp
  absorption_partial.hpp
  emission_partial.hpp
  volume_partial.hpp
  utils/file_utils.hpp
  utils/png_compare.hpp
  utils/png_decoder.hpp
  utils/png_encoder.hpp
  utils/lodepng.h
  )

set(apcomp_sources
  apcomp.cpp
  image.cpp
  compositor.cpp
  partial_compositor.cpp
  scalar_compositor.cpp
  utils/file_utils.cpp
  utils/png_compare.cpp
  utils/png_decoder.cpp
  utils/png_encoder.cpp
  utils/lodepng.cpp
  )


#------------------------------------------------------------------------------
# serial (non mpi) apcomp lib
#------------------------------------------------------------------------------
if (ENABLE_SERIAL)
    set(apcomp_deps )

    if(ENABLE_OPENMP)
        list(APPEND apcomp_deps ${ascent_blt_openmp_deps})
    endif()

    blt_add_library(NAME apcomp
                    SOURCES ${apcomp_sources}
                    HEADERS ${apcomp_headers}
                    DEPENDS_ON ${apcomp_deps}
                    )

    # extra defs and props
    target_compile_definitions(apcomp PRIVATE ASCENT_EXPORTS_FLAG)
    if(ENABLE_OPENMP)
        target_compile_definitions(apcomp PRIVATE APCOMP_USE_OPENMP)
    endif()
    set_target_properties(apcomp PROPERTIES CXX_VISIBILITY_PRESET hidden)

    # build includes
    # this allows us to include as <apcomp/header.hpp>
    target_include_directories(apcomp PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(apcomp PUBLIC $<INSTALL_INTERFACE:include/ascent/apcomp>)

    # install target
    install(TARGETS apcomp
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )

    install(FILES ${apcomp_headers} DESTINATION include/ascent/apcomp)

endif()

#------------------------------------------------------------------------------
# parallel apcomp_mpi lib
#------------------------------------------------------------------------------
if (MPI_FOUND)
    set(apcomp_mpi_headers
      internal/DirectSendCompositor.hpp
      internal/MPICollect.hpp
      internal/RadixKCompositor.hpp
      internal/apcomp_diy_collect.hpp
      internal/apcomp_diy_image_block.hpp
      internal/apcomp_diy_utils.hpp
      )

    list(APPEND apcomp_mpi_headers ${apcomp_headers})

    set(apcomp_mpi_sources
      internal/DirectSendCompositor.cpp
      internal/RadixKCompositor.cpp
      )

    list(APPEND apcomp_mpi_sources ${apcomp_sources})

    set(apcomp_mpi_deps ${ascent_blt_mpi_deps})

    if(ENABLE_OPENMP)
         list(APPEND apcomp_mpi_deps ${ascent_blt_openmp_deps})
    endif()


    blt_add_library(
        NAME apcomp_mpi
        SOURCES ${apcomp_sources} ${apcomp_mpi_sources}
        HEADERS ${apcomp_headers} ${apcomp_mpi_headers}
        DEPENDS_ON ${apcomp_mpi_deps}
    )

    # build includes
    # this allows us to include as <apcomp/header.hpp>
    target_include_directories(apcomp_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(apcomp_mpi PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/diy/include/>)
    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(apcomp_mpi PUBLIC $<INSTALL_INTERFACE:include/ascent/>)

    # extra defs and props
    target_compile_definitions(apcomp_mpi PRIVATE APCOMP_PARALLEL)
    target_compile_definitions(apcomp_mpi PRIVATE ASCENT_EXPORTS_FLAG)
    if(ENABLE_OPENMP)
      target_compile_definitions(apcomp_mpi PRIVATE APCOMP_USE_OPENMP)
    endif()


    # install target
    install(TARGETS apcomp_mpi
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )

    install(FILES  ${apcomp_mpi_headers} DESTINATION include/ascent/apcomp)

endif()
