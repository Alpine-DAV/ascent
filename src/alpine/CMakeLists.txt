###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Alpine. 
# 
# For details, see: http://software.llnl.gov/alpine/.
# 
# Please also read alpine/LICENSE
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
# file: src/alpine/CMakeLists.txt
#
###############################################################################

set(ALPINE_WEB_CLIENT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/web_client)

set(ALPINE_VTKM_ENABLED ${VTKM_FOUND})
set(ALPINE_VTKH_ENABLED ${VTKM_FOUND})

set(ALPINE_HDF5_ENABLED ${HDF5_FOUND})

if(ALPINE_VTKM_ENABLED)
    set(ALPINE_VTKM_USE_CUDA ${CUDA_FOUND})
    set(ALPINE_VTKM_USE_TBB  ${TBB_FOUND})
endif()


set(ALPINE_USE_OPENMP  ${OPENMP_FOUND})

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/alpine_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/alpine_config.h")

set(alpine_sources
    # alpine interface
    alpine.cpp
    c/alpine_c.cpp
    # runtimes
    alpine_runtime.cpp
    runtimes/alpine_empty_runtime.cpp
    runtimes/alpine_flow_runtime.cpp
    runtimes/flow_filters/alpine_runtime_filters.cpp
    runtimes/flow_filters/alpine_runtime_relay_filters.cpp
    runtimes/flow_filters/alpine_runtime_blueprint_filters.cpp
    # utils
    utils/alpine_file_system.cpp
    utils/alpine_block_timer.cpp
    utils/alpine_png_encoder.cpp
    utils/alpine_web_interface.cpp)

set(alpine_headers
    # alpine interface
    ${CMAKE_CURRENT_BINARY_DIR}/alpine_config.h
    alpine_exports.h
    alpine.hpp
    c/alpine.h
    # runtimes
    alpine_runtime.hpp
    runtimes/alpine_empty_runtime.hpp
    # flow
    runtimes/alpine_flow_runtime.hpp
    runtimes/flow_filters/alpine_runtime_filters.hpp
    runtimes/flow_filters/alpine_runtime_relay_filters.hpp
    runtimes/flow_filters/alpine_runtime_blueprint_filters.hpp
    # utils
    utils/alpine_logging.hpp
    utils/alpine_file_system.hpp
    utils/alpine_block_timer.hpp
    utils/alpine_png_encoder.hpp
    utils/alpine_web_interface.hpp)

if(VTKM_FOUND)

    # data adapter
    list(APPEND alpine_headers runtimes/alpine_data_adapter.hpp)
    list(APPEND alpine_sources runtimes/alpine_data_adapter.cpp)

    # VTKm Pipeline Headers
    list(APPEND alpine_headers runtimes/alpine_vtkm_runtime.hpp)
    list(APPEND alpine_headers runtimes/alpine_vtkm_renderer.hpp)

    # VTKm Pipeline Sources
    list(APPEND alpine_sources runtimes/alpine_vtkm_runtime.cpp)
    # VTKm Rendering Headers
    list(APPEND alpine_headers runtimes/alpine_vtkm_renderer.hpp)

    # VTKm Rendering Sources
    list(APPEND alpine_sources runtimes/alpine_vtkm_renderer.cpp)

    # vtkh filters
    list(APPEND alpine_headers runtimes/flow_filters/alpine_runtime_vtkh_filters.hpp)
    list(APPEND alpine_sources runtimes/flow_filters/alpine_runtime_vtkh_filters.cpp)


    # main pipeline
    list(APPEND alpine_headers runtimes/alpine_main_runtime.hpp)
    list(APPEND alpine_sources runtimes/alpine_main_runtime.cpp)

endif()

##############################
# optional features
#############################

if(FORTRAN_FOUND)
    list(APPEND alpine_sources fortran/alpine_fortran.f90)
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

set(alpine_thirdparty_libs 
    conduit
    flow
    lodepng)

##################
# optional libs
##################
if(VTKM_FOUND)
    list(APPEND alpine_thirdparty_libs vtkm vtkh)
endif()

if(CUDA_FOUND)
    list(APPEND alpine_thirdparty_libs cuda_runtime)
endif()


if(TBB_FOUND)
    list(APPEND alpine_thirdparty_libs tbb)
endif()


##########################################
# Build a serial version of alpine
##########################################

blt_add_library(
    NAME        alpine
    SOURCES     ${alpine_sources} 
    HEADERS     ${alpine_headers}
    DEPENDS_ON  ${alpine_thirdparty_libs})

# install target for serial alpine lib
install(TARGETS alpine
        EXPORT  alpine
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

################################################
# Build Parallel (MPI) version of alpine
################################################
if(MPI_FOUND)
    if(VTKM_FOUND)
        # we only need compositing if we are rendering with vtk-m
        set(alpine_par_sources utils/alpine_icet_compositor.cpp
                               utils/alpine_diy_compositor.cpp
                               utils/diy_compositing/alpine_diy_direct_send.cpp
                               utils/diy_compositing/alpine_diy_direct_send.cpp
                               utils/diy_compositing/alpine_diy_radix_k.cpp)
                     
        set(alpine_par_headers utils/alpine_icet_compositor.hpp
                               utils/alpine_compositor_base.hpp
                               utils/alpine_diy_compositor.hpp
                               utils/diy_compositing/alpine_diy_image.hpp
                               utils/diy_compositing/alpine_diy_collect.hpp
                               utils/diy_compositing/alpine_diy_direct_send.hpp
                               utils/diy_compositing/alpine_diy_radix_k.hpp)
    endif()
   
    # req'd libs 
    set(alpine_par_thirdparty_libs mpi conduit_relay_mpi)

    # optional libs
    if(ICET_FOUND)
        list(APPEND alpine_par_thirdparty_libs icet)
    endif()

    list(APPEND alpine_par_thirdparty_libs ${alpine_thirdparty_libs})

    if(VTKM_FOUND)
        # swap vtkh_par for vtkh
        list(REMOVE_ITEM alpine_par_thirdparty_libs vtkh)
        list(APPEND alpine_par_thirdparty_libs vtkh_par)
    endif()

    blt_add_library(
        NAME        alpine_par
        SOURCES     ${alpine_sources} ${alpine_par_sources}
        HEADERS     ${alpine_headers} ${alpine_par_headers}
        DEPENDS_ON  ${alpine_par_thirdparty_libs})
    
    blt_add_target_compile_flags(TO alpine_par FLAGS "-D PARALLEL")

    # install target for mpi ver
    install(TARGETS alpine_par
            EXPORT  alpine
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )

endif()

########################################
# Extra Install Targets for alpine
########################################

install(FILES ${alpine_headers} DESTINATION include/alpine)


