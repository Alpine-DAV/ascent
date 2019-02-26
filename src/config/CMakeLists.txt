###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################


###############################################################################
#
# file: src/config/CMakeLists.txt
#
###############################################################################


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
endif(endif)

# static builds require python 
set(ASCENT_CONFIG_MK_PYTHON_LINK_FLAGS "")
if(PYTHON_FOUND)
    foreach(_lib ${PYTHON_LIBRARY})
        set(ASCENT_CONFIG_MK_PYTHON_LINK_FLAGS "${ASCENT_CONFIG_MK_PYTHON_LINK_FLAGS} ${_lib}")
    endforeach()
endif(endif)


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/ascent_config.mk.in"
                "${CMAKE_CURRENT_BINARY_DIR}/ascent_config.mk")


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
  set(ASCENT_INSTALL_CMAKE_MODULE_DIR "${ASCENT_INSTALL_CONFIG_DIR}/cmake")
endif()

include(CMakePackageConfigHelpers)
# setup cmake exports
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/ascent-config.cmake.in
  ${PROJECT_BINARY_DIR}/${ASCENT_INSTALL_CMAKE_MODULE_DIR}/ascent-config.cmake
  INSTALL_DESTINATION ${ASCENT_INSTALL_CONFIG_DIR}
  PATH_VARS
    ASCENT_INSTALL_INCLUDE_DIR
    ASCENT_INSTALL_LIB_DIR
    ASCENT_INSTALL_BIN_DIR
    ASCENT_INSTALL_SHARED_RESOURCES_DIR
    ASCENT_INSTALL_CMAKE_MODULE_DIR
    )

configure_file (${PROJECT_SOURCE_DIR}/cmake/ascent-deps-paths.cmake.in 
                ${PROJECT_BINARY_DIR}/${ASCENT_INSTALL_CMAKE_MODULE_DIR}/ascent-deps-paths.cmake @ONLY)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ascent_config.mk"
        DESTINATION share/ascent/)

install(FILES
        ${PROJECT_BINARY_DIR}/${ASCENT_INSTALL_CMAKE_MODULE_DIR}/ascent-config.cmake
        ${PROJECT_BINARY_DIR}/${ASCENT_INSTALL_CMAKE_MODULE_DIR}/ascent-deps-paths.cmake
        DESTINATION ${ASCENT_INSTALL_CMAKE_MODULE_DIR})



install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ascent_config.mk" DESTINATION share/ascent)



