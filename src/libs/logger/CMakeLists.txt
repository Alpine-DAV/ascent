# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/logger_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/logger_config.h")


set(logger_headers
    Logger.hpp
  )

set(logger_sources
    Logger.cpp
  )


install(FILES ${logger_headers} DESTINATION include/ascent/logger)

#------------------------------------------------------------------------------
# serial (non mpi) logger lib
#------------------------------------------------------------------------------
if (ENABLE_SERIAL)
    set(logger_deps conduit::conduit ascent_png_utils)

    if(ENABLE_OPENMP)
        list(APPEND logger_deps ${ascent_blt_openmp_deps})
    endif()

    blt_add_library(NAME logger
                    SOURCES ${logger_sources}
                    HEADERS ${logger_headers}
                    DEPENDS_ON ${logger_deps}
                    )

    # extra defs and props
    target_compile_definitions(logger PRIVATE ASCENT_EXPORTS_FLAG)

    if(ENABLE_HIDDEN_VISIBILITY)
        set_target_properties(logger PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

    # build includes
    # this allows us to include as <logger/header.hpp>
    target_include_directories(logger PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(logger PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)
    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(logger PUBLIC $<INSTALL_INTERFACE:include/ascent/logger>)

    # install target
    install(TARGETS logger
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )



endif()

#------------------------------------------------------------------------------
# parallel logger_mpi lib
#------------------------------------------------------------------------------
if (MPI_FOUND)

    set(logger_mpi_headers ${logger_headers})
    # install extra mpi related headers
    install(FILES  ${logger_mpi_headers} DESTINATION include/ascent/logger)

    set(logger_mpi_sources ${logger_sources})

    set(logger_mpi_deps conduit::conduit ascent_png_utils ${ascent_blt_mpi_deps})

    if(ENABLE_OPENMP)
         list(APPEND logger_mpi_deps ${ascent_blt_openmp_deps})
    endif()


    blt_add_library(NAME logger_mpi
                    SOURCES ${logger_sources} ${logger_mpi_sources}
                    HEADERS ${logger_headers} ${logger_mpi_headers}
                    DEPENDS_ON ${logger_mpi_deps}
                    )

    # build includes
    # this allows us to include as <logger/header.hpp>
    target_include_directories(logger_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(logger_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)
    target_include_directories(logger_mpi PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/diy/include/>)
    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(logger_mpi PUBLIC $<INSTALL_INTERFACE:include/ascent/logger>)

    # extra defs and props
    target_compile_definitions(logger_mpi PRIVATE LOGGER_PARALLEL)
    target_compile_definitions(logger_mpi PRIVATE ASCENT_EXPORTS_FLAG)

    if(ENABLE_HIDDEN_VISIBILITY)
        set_target_properties(logger_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

    # install target
    install(TARGETS logger_mpi
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )

endif()
