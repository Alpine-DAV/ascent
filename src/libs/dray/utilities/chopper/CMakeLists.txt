# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

set(chopper_thirdparty_libs dray)
set(chopper_thirdparty_libs_mpi dray_mpi)

################################################
# chopper
################################################
if(ENABLE_SERIAL)
  blt_add_executable(NAME chopper
                     SOURCES chopper.cpp
                     DEPENDS_ON ${chopper_thirdparty_libs}
                     OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                     )
endif()

################################################
# MPI SUPPORT
################################################

if(MPI_FOUND)
################################################
# chopper mpi
################################################
  blt_add_executable(NAME chopper_mpi
                     SOURCES chopper.cpp
                     DEPENDS_ON ${chopper_thirdparty_libs_mpi}
                     OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                    )

  target_compile_definitions(chopper_mpi PRIVATE "MPI_ENABLED")

endif()

configure_file(chopper_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/chopper_config.yaml @ONLY)
