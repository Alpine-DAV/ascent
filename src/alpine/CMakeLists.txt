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
set(ALPINE_HDF5_ENABLED ${HDF5_FOUND})

if(ALPINE_VTKM_ENABLED)
    set(ALPINE_VTKM_USE_CUDA ${CUDA_FOUND})
    set(ALPINE_VTKM_USE_TBB  ${TBB_FOUND})
endif()



configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/alpine_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/alpine_config.h")

set(alpine_sources
    # alpine interface
    alpine.cpp
    c/alpine_c.cpp
    # pipelines
    alpine_pipeline.cpp
    pipelines/alpine_empty_pipeline.cpp
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
    # pipelines
    alpine_pipeline.hpp
    pipelines/alpine_empty_pipeline.hpp
    # utils
    utils/alpine_logging.hpp
    utils/alpine_file_system.hpp
    utils/alpine_block_timer.hpp
    utils/alpine_png_encoder.hpp
    utils/alpine_web_interface.hpp)

if(VTKM_FOUND)
    # VTKm Pipeline Headers
    list(APPEND alpine_headers pipelines/alpine_vtkm_pipeline.hpp)
    list(APPEND alpine_headers pipelines/alpine_vtkm_renderer.hpp)


    # VTKm Pipeline Sources
    list(APPEND alpine_sources pipelines/alpine_vtkm_pipeline.cpp)
    # VTKm Rendering Headers
    list(APPEND alpine_headers pipelines/alpine_vtkm_renderer.hpp)

    # VTKm Rendering Sources
    list(APPEND alpine_sources pipelines/alpine_vtkm_renderer.cpp)
    
   
endif()

if(HDF5_FOUND)
    list(APPEND alpine_headers pipelines/alpine_blueprint_hdf5_pipeline.hpp)
    list(APPEND alpine_sources pipelines/alpine_blueprint_hdf5_pipeline.cpp)
endif()

##############################
# Add lodepng 
#############################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty_builtin/lodepng/ ${CMAKE_CURRENT_SOURCE_DIR}/utils/lodepng)

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
    conduit_relay
    conduit_blueprint)

if(VTKM_FOUND)
    list(APPEND alpine_thirdparty_libs vtkm)
    list(APPEND alpine_thirdparty_libs vtkm_cont)
    list(APPEND alpine_thirdparty_libs vtkm_rendering)
endif()

##########################################
# Build a serial version of alpine
##########################################

if(BUILD_SHARED_LIBS)
    message(STATUS "Building alpine as a shared library")
     add_library(alpine SHARED 
                 ${alpine_sources} 
                 $<TARGET_OBJECTS:lodepng>
                 ${alpine_headers})
else()
    message(STATUS "Building alpine as a static library")
    add_library(alpine STATIC 
                ${alpine_sources} 
                 $<TARGET_OBJECTS:lodepng>
                ${alpine_headers})
endif()

##########################################
# Ensure we use c++11
##########################################

set_target_properties(alpine PROPERTIES
                      CXX_STANDARD 11
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENTIONS NO)

add_target_compile_flags(TARGET alpine 
                         FLAGS ${VTKm_COMPILE_OPTIONS})


target_link_libraries(alpine ${alpine_thirdparty_libs})

if(CUDA_FOUND)
    target_link_libraries(alpine ${CUDA_CUDART_LIBRARY})
endif()


if(TBB_FOUND)
   target_link_libraries(alpine ${TBB_LIBRARIES})
endif()


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
    set(alpine_par_sources utils/alpine_diy_compositor.cpp)
    set(alpine_par_sources utils/alpine_icet_compositor.cpp)
    set(alpine_par_headers utils/alpine_diy_compositor.hpp)
    set(alpine_par_headers utils/alpine_icet_compositor.hpp)
    

    if(VTKM_FOUND)
        # we need to set compiler flags for the MPI case
        # we have to global cxx flags b/c of how the cmake cuda wrappers work 
        # we save / restore the previous set of global flags after we create the cuda 
        # target
        SET(CMAKE_CXX_FLAGS_PREVIOUS ${CMAKE_CXX_FLAGS})
        SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} -D PARALLEL -fPIC" )
        
        # restore old global cxx flags
        SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_PREVIOUS})
    endif()
   
    if(BUILD_SHARED_LIBS)
        message(STATUS "Building alpine_par as a shared library")
        add_library(alpine_par SHARED ${alpine_sources} 
                                      ${alpine_par_sources} 
                                      ${alpine_headers}
                                      $<TARGET_OBJECTS:lodepng>
                                      ${alpine_par_headers})
    else()
        message(STATUS "Building alpine_par as a static library")
        add_library(alpine_par STATIC ${alpine_sources} 
                                      ${alpine_par_sources} 
                                      ${alpine_headers}
                                      $<TARGET_OBJECTS:lodepng>
                                      ${alpine_par_headers})
    endif()
    
    ##########################################
    # Ensure we use c++11
    ##########################################

    set_target_properties(alpine_par PROPERTIES
                          CXX_STANDARD 11
                          CXX_STANDARD_REQUIRED YES
                          CXX_EXTENTIONS NO)

    add_target_compile_flags(TARGET alpine_par 
                             FLAGS "${MPI_CXX_COMPILE_FLAGS} -D PARALLEL ${VTKm_COMPILE_OPTIONS}")
        
    add_target_link_flags(TARGET alpine_par  
                          FLAGS "${MPI_CXX_LINK_FLAGS}")
    
    target_link_libraries(alpine_par
                          conduit_relay_mpi
                          ${alpine_thirdparty_libs}
                          ${ICET_LIBRARIES}
                          ${MPI_CXX_LIBRARIES})


   
    if(CUDA_FOUND)
        target_link_libraries(alpine_par ${CUDA_CUDART_LIBRARY})
    endif()

    if(TBB_FOUND)
       target_link_libraries(alpine_par ${TBB_LIBRARIES})
    endif()
  
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


