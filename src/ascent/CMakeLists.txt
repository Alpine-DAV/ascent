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
# file: src/ascent/CMakeLists.txt
#
###############################################################################

convert_to_native_escaped_file_path(${CMAKE_INSTALL_PREFIX}
                                    ASCENT_INSTALL_PREFIX)

set(ASCENT_INSTALL_PREFIX ${ASCENT_INSTALL_PREFIX} CACHE STRING "" FORCE )

if(FORTRAN_FOUND)
    set(ASCENT_FORTRAN_COMPILER ${CMAKE_Fortran_COMPILER})
endif()

set(ASCENT_WEB_CLIENT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/web_client)

set(ASCENT_VTKM_ENABLED ${VTKM_FOUND})
set(ASCENT_VTKH_ENABLED ${VTKM_FOUND})

set(ASCENT_HDF5_ENABLED ${HDF5_FOUND})
set(ASCENT_ADIOS_ENABLED ${ADIOS_FOUND})
set(ASCENT_MFEM_ENABLED ${MFEM_FOUND})

if(ASCENT_VTKM_ENABLED)
    set(ASCENT_VTKM_USE_CUDA ${CUDA_FOUND})
    set(ASCENT_VTKM_USE_OPENMP ${ENABLE_OPENMP})
endif()


set(ASCENT_USE_OPENMP  ${OPENMP_FOUND})

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/ascent_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/ascent_config.h")

set(ascent_sources
    # ascent interface
    ascent.cpp
    c/ascent_c.cpp
    # runtimes
    ascent_runtime.cpp
    runtimes/ascent_main_runtime.cpp
    runtimes/ascent_empty_runtime.cpp
    runtimes/ascent_flow_runtime.cpp
    runtimes/flow_filters/ascent_runtime_filters.cpp
    runtimes/flow_filters/ascent_runtime_param_check.cpp
    runtimes/flow_filters/ascent_runtime_relay_filters.cpp
    runtimes/flow_filters/ascent_runtime_blueprint_filters.cpp
    # utils
    utils/ascent_file_system.cpp
    utils/ascent_block_timer.cpp
    utils/ascent_png_compare.cpp
    utils/ascent_png_decoder.cpp
    utils/ascent_png_encoder.cpp
    utils/ascent_string_utils.cpp
    utils/ascent_web_interface.cpp
    # hola
    hola/ascent_hola.cpp)

set(ascent_headers
    # ascent interface
    ${CMAKE_CURRENT_BINARY_DIR}/ascent_config.h
    ascent_exports.h
    ascent.hpp
    c/ascent.h
    # runtimes
    ascent_runtime.hpp
    runtimes/ascent_empty_runtime.hpp
    # flow
    runtimes/ascent_main_runtime.hpp
    runtimes/ascent_flow_runtime.hpp
    runtimes/flow_filters/ascent_runtime_filters.hpp
    runtimes/flow_filters/ascent_runtime_param_check.hpp
    runtimes/flow_filters/ascent_runtime_relay_filters.hpp
    runtimes/flow_filters/ascent_runtime_blueprint_filters.hpp
    # utils
    utils/ascent_logging.hpp
    utils/ascent_file_system.hpp
    utils/ascent_block_timer.hpp
    utils/ascent_png_compare.hpp
    utils/ascent_png_decoder.hpp
    utils/ascent_png_encoder.hpp
    utils/ascent_string_utils.hpp
    utils/ascent_web_interface.hpp
    # hola
    hola/ascent_hola.hpp)

if(VTKM_FOUND)

    # data adapter
    list(APPEND ascent_headers runtimes/ascent_vtkh_data_adapter.hpp)
    list(APPEND ascent_sources runtimes/ascent_vtkh_data_adapter.cpp)

    # vtkh filters
    list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp)
    list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_vtkh_filters.cpp)

    # rover
    list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_rover_filters.hpp)
    list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_rover_filters.cpp)

    # parsing utils
    list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_conduit_to_vtkm_parsing.hpp)
    list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_conduit_to_vtkm_parsing.cpp)

    # main runtime
    list(APPEND ascent_headers runtimes/ascent_main_runtime.hpp)
    list(APPEND ascent_sources runtimes/ascent_main_runtime.cpp)

endif()

if(MFEM_FOUND)
    # data adapter
    list(APPEND ascent_headers runtimes/ascent_mfem_data_adapter.hpp)
    list(APPEND ascent_sources runtimes/ascent_mfem_data_adapter.cpp)
endif()
##############################
# optional features
#############################

if(FORTRAN_FOUND)
    add_library(ascent_fortran OBJECT fortran/ascent_fortran.f90)
    list(APPEND ascent_sources $<TARGET_OBJECTS:ascent_fortran>)
endif()

################################
# Add python wrappers if python
# support was selected
################################
if(PYTHON_FOUND)
    add_subdirectory(python)
endif()

##################
# req'd libs
##################

set(ascent_thirdparty_libs
    conduit
    flow
    lodepng)

##################
# optional libs
##################
if(VTKM_FOUND)
    list(APPEND ascent_thirdparty_libs vtkm vtkh rover)
endif()

if(MFEM_FOUND)
    list(APPEND ascent_thirdparty_libs mfem)
    # workaround for MPI headers
    if(ENABLE_MPI)
        list(APPEND ascent_thirdparty_libs mpi)
    endif()
endif()

if(CUDA_FOUND)
    list(APPEND ascent_thirdparty_libs cuda_runtime)
endif()

if(ENABLE_OPENMP)
    list(APPEND ascent_thirdparty_libs openmp)
endif()

##########################################
# Build a serial version of ascent
##########################################

blt_add_library(
    NAME        ascent
    SOURCES     ${ascent_sources}
    HEADERS     ${ascent_headers}
    DEPENDS_ON  ${ascent_thirdparty_libs})

# install target for serial ascent lib
install(TARGETS ascent
        EXPORT  ascent
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

################################################
# Build Parallel (MPI) version of ascent
################################################
if(MPI_FOUND)

    # req'd libs
    set(ascent_mpi_thirdparty_libs mpi conduit_relay_mpi)

    # hola mpi support is only in par (mpi ver)

    list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_hola_filters.hpp)
    list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_hola_filters.cpp)

    list(APPEND ascent_headers hola/ascent_hola_mpi.hpp)
    list(APPEND ascent_sources hola/ascent_hola_mpi.cpp)

    # adios support is only in par (mpi ver)
    if (ADIOS_FOUND)
       list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_adios_filters.hpp)
       list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_adios_filters.cpp)
       list(APPEND ascent_mpi_thirdparty_libs adios)
    endif()

    list(APPEND ascent_mpi_thirdparty_libs ${ascent_thirdparty_libs})

    if(VTKM_FOUND)
        # swap vtkh_mpi for vtkh
        list(REMOVE_ITEM ascent_mpi_thirdparty_libs vtkh)
        list(APPEND ascent_mpi_thirdparty_libs vtkh_mpi)
        list(REMOVE_ITEM ascent_mpi_thirdparty_libs rover)
        list(APPEND ascent_mpi_thirdparty_libs rover_mpi)
    endif()

    blt_add_library(
        NAME        ascent_mpi
        SOURCES     ${ascent_sources} ${ascent_mpi_sources}
        HEADERS     ${ascent_headers} ${ascent_mpi_headers}
        DEPENDS_ON  ${ascent_mpi_thirdparty_libs})

    blt_add_target_compile_flags(TO ascent_mpi FLAGS "-D ASCENT_MPI_ENABLED")

    # install target for mpi ver
    install(TARGETS ascent_mpi
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )

endif()

########################################
# Extra Install Targets for ascent
########################################
install(FILES ${ascent_headers} DESTINATION include/ascent)


