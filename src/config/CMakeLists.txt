###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


###############################################################################
#
# file: src/config/CMakeLists.txt
#
###############################################################################

###############################################################################
# create config.mk for makefile-based build systems
###############################################################################

# extra sauce for needed for config.mk

# special handling for mpi libs, due to possible transitive mpi dep from mfem
set(ASCENT_CONFIG_MK_MPI_LINK_FLAGS "")
if(MPI_FOUND)
    foreach(_lib ${MPI_C_LIBRARIES})
        set(ASCENT_CONFIG_MK_MPI_LINK_FLAGS "${ASCENT_CONFIG_MK_MPI_LINK_FLAGS} ${_lib}")
    endforeach()

    foreach(_lib ${MPI_CXX_LIBRARIES})
        set(ASCENT_CONFIG_MK_MPI_LINK_FLAGS "${ASCENT_CONFIG_MK_MPI_LINK_FLAGS} ${_lib}")
    endforeach()

    foreach(_lib ${MPI_Fortran_LIBRARIES})
        set(ASCENT_CONFIG_MK_MPI_LINK_FLAGS "${ASCENT_CONFIG_MK_MPI_LINK_FLAGS} ${_lib}")
    endforeach()
endif()

# static builds require python 
set(ASCENT_CONFIG_MK_PYTHON_LINK_FLAGS "")
if(PYTHON_FOUND)
    set(ASCENT_CONFIG_MK_PYTHON_LIB_DIR ${PYTHON_LIB_DIR})
    foreach(_lib ${PYTHON_LIBRARY})
        set(ASCENT_CONFIG_MK_PYTHON_LINK_FLAGS "${ASCENT_CONFIG_MK_PYTHON_LINK_FLAGS} ${_lib}")
    endforeach()
endif()

# builtin libs vars for config.mk gen
if(ENABLE_APCOMP)
    set(ASCENT_APCOMP_ENABLED "TRUE")
endif()

if(ENABLE_DRAY)
    set(ASCENT_DRAY_ENABLED "TRUE")
endif()


# generate the config.mk
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/ascent_config.mk.in"
                "${CMAKE_CURRENT_BINARY_DIR}/ascent_config.mk")

# install the config.mk
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ascent_config.mk" DESTINATION share/ascent)



# install optional libs config info
if(ENABLE_APCOMP)
    # generate the config.mk
    configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/apcomp_config.mk.in"
                    "${CMAKE_CURRENT_BINARY_DIR}/apcomp_config.mk")
    # install the config.mk
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/apcomp_config.mk" DESTINATION share/ascent)
endif()

if(ENABLE_DRAY)
    # generate the config.mk
    configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/dray_config.mk.in"
                    "${CMAKE_CURRENT_BINARY_DIR}/dray_config.mk")
    # install the config.mk
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dray_config.mk" DESTINATION share/ascent)
endif()


###############################################################################
# export everything for a cmake build to be able to import with find_package
###############################################################################

if (NOT DEFINED ASCENT_INSTALL_INCLUDE_DIR)
  set(ASCENT_INSTALL_INCLUDE_DIR "include")
endif()

if (NOT DEFINED ASCENT_INSTALL_CONFIG_DIR)
  set(ASCENT_INSTALL_CONFIG_DIR "lib")
endif()

if (NOT DEFINED ASCENT_INSTALL_LIB_DIR)
  set(ASCENT_INSTALL_LIB_DIR "lib")
endif()

if (NOT DEFINED ASCENT_INSTALL_BIN_DIR)
  set(ASCENT_INSTALL_BIN_DIR "bin")
endif()

if (NOT DEFINED ASCENT_INSTALL_SHARED_RESOURCES_DIR)
  set(ASCENT_INSTALL_SHARED_RESOURCES_DIR "share/ascent")
endif()

if (NOT DEFINED ASCENT_INSTALL_CMAKE_MODULE_DIR)
  if(WIN32)
    set(ASCENT_INSTALL_CMAKE_MODULE_DIR "cmake")
  else()
    set(ASCENT_INSTALL_CMAKE_MODULE_DIR "${ASCENT_INSTALL_CONFIG_DIR}/cmake/ascent")
  endif()
endif()

include(CMakePackageConfigHelpers)

# write version heler
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/AscentConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)


# write cmake package config file
configure_package_config_file(
  AscentConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/AscentConfig.cmake
  INSTALL_DESTINATION ${ASCENT_INSTALL_CMAKE_MODULE_DIR}
  PATH_VARS
    ASCENT_INSTALL_INCLUDE_DIR
    ASCENT_INSTALL_LIB_DIR
    ASCENT_INSTALL_BIN_DIR
    ASCENT_INSTALL_SHARED_RESOURCES_DIR
    ASCENT_INSTALL_CMAKE_MODULE_DIR
    )


# install everything needed
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/AscentConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/AscentConfigVersion.cmake
    ascent_setup_deps.cmake
    ascent_setup_targets.cmake
    DESTINATION ${ASCENT_INSTALL_CMAKE_MODULE_DIR})






