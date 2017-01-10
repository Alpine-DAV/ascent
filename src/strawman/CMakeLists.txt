###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Strawman. 
# 
# For details, see: http://software.llnl.gov/strawman/.
# 
# Please also read strawman/LICENSE
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
# file: src/strawman/CMakeLists.txt
#
###############################################################################

set(STRAWMAN_WEB_CLIENT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/web_client)

set(STRAWMAN_EAVL_ENABLED ${EAVL_FOUND})
set(STRAWMAN_VTKM_ENABLED ${VTKM_FOUND})
set(STRAWMAN_HDF5_ENABLED ${HDF5_FOUND})

if(STRAWMAN_EAVL_ENABLED)
    set(STRAWMAN_VTKM_USE_OPENMP ${OPENMP_FOUND})
    set(STRAWMAN_EAVL_USE_CUDA   ${CUDA_FOUND})
endif()


if(STRAWMAN_VTKM_ENABLED)
    set(STRAWMAN_VTKM_USE_CUDA ${CUDA_FOUND})
    set(STRAWMAN_VTKM_USE_TBB  ${TBB_FOUND})
endif()



configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/strawman_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/strawman_config.h")

set(strawman_sources
    # strawman interface
    strawman.cpp
    c/strawman_c.cpp
    # pipelines
    strawman_pipeline.cpp
    pipelines/strawman_empty_pipeline.cpp
    # utils
    utils/strawman_file_system.cpp
    utils/strawman_block_timer.cpp
    utils/strawman_png_encoder.cpp
    utils/strawman_web_interface.cpp
    )


set(strawman_headers
    # strawman interface
    ${CMAKE_CURRENT_BINARY_DIR}/strawman_config.h
    strawman_exports.h
    strawman.hpp
    c/strawman.h
    # pipelines
    strawman_pipeline.hpp
    pipelines/strawman_empty_pipeline.hpp
    # utils
    utils/strawman_logging.hpp
    utils/strawman_file_system.hpp
    utils/strawman_block_timer.hpp
    utils/strawman_png_encoder.hpp
    utils/strawman_web_interface.hpp
    )

if(EAVL_FOUND)
    #
    # EAVL Pipeline Headers
    #
    list(APPEND strawman_headers pipelines/strawman_eavl_pipeline.hpp)

    #
    # EAVL Pipeline Sources
    #
    list(APPEND strawman_sources pipelines/strawman_eavl_pipeline.cpp)

endif()

if(VTKM_FOUND)
    set(strawman_vtkm_sources "")
    set(strawman_vtkm_headers "")
    
    # VTKm Pipeline Headers
    list(APPEND strawman_vtkm_headers pipelines/strawman_vtkm_pipeline.hpp)
    list(APPEND strawman_vtkm_headers pipelines/strawman_vtkm_pipeline_backend.hpp)
    list(APPEND strawman_vtkm_headers pipelines/strawman_vtkm_renderer.hpp)


    # VTKm Pipeline Sources
    list(APPEND strawman_vtkm_sources pipelines/strawman_vtkm_pipeline.cpp)
    list(APPEND strawman_vtkm_sources pipelines/strawman_vtkm_pipeline_backend_serial.cpp)


    # VTKm Rendering Headers
    set(strawman_vtkm_renderer_headers pipelines/strawman_vtkm_renderer.hpp)

    # VTKm Rendering Sources
    set(strawman_vtkm_renderer_sources pipelines/strawman_vtkm_renderer_serial.cpp)
    
    if(TBB_FOUND)
       list(APPEND strawman_vtkm_sources pipelines/strawman_vtkm_pipeline_backend_tbb.cpp)
       list(APPEND strawman_vtkm_renderer_sources pipelines/strawman_vtkm_renderer_tbb.cpp)
    endif()
    
    #
    # VTKm library objects
    #
    add_library(strawman_vtkm_renderer OBJECT ${strawman_vtkm_renderer_sources} ${strawman_vtkm_renderer_headers})
    add_library(strawman_vtkm OBJECT ${strawman_vtkm_sources} ${strawman_vtkm_headers})
        

    if(CUDA_FOUND)
        #
        # VTKm cuda objects
        #
        
        # cuda_wrap_srcs doesn't seem to pick up -fPIC, even though we have 
        # set(CMAKE_POSITION_INDEPENDENT_CODE ON) enabled for the entire project
        # to add fPIC, we have to global cxx flags b/c of how the cmake cuda wrappers work 
        # we save / restore the previous set of global flags after we create the cuda 
        # target

        SET(CMAKE_CXX_FLAGS_PREVIOUS ${CMAKE_CXX_FLAGS})
        SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC" )

        cuda_wrap_srcs(strawman_vtkm_pipeline_cuda OBJ vtkm_cuda_pipeline pipelines/strawman_vtkm_pipeline_backend_cuda.cu)
        cuda_wrap_srcs(strawman_vtkm_renderer_cuda OBJ vtkm_cuda_renderer pipelines/strawman_vtkm_renderer_cuda.cu)
        
        # restore old global cxx flags
        SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_PREVIOUS})

    endif()

endif()

if(HDF5_FOUND)
    list(APPEND strawman_headers pipelines/strawman_blueprint_hdf5_pipeline.hpp)
    list(APPEND strawman_sources pipelines/strawman_blueprint_hdf5_pipeline.cpp)
endif()



##############################
# optional features
#############################

if(FORTRAN_FOUND)
    list(APPEND strawman_sources fortran/strawman_fortran.f90)
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

set(strawman_thirdparty_libs 
    conduit
    conduit_relay
    conduit_blueprint
    lodepng)

if(EAVL_FOUND)
    list(APPEND strawman_thirdparty_libs
                ${EAVL_LIBRARIES}
                ${OSMESA_LIBRARIES})

endif()


##########################################
# Build a serial version of strawman
##########################################



if(BUILD_SHARED_LIBS)
    message(STATUS "Building strawman as a shared library")
    if(VTKM_FOUND)
        add_library(strawman SHARED 
                    ${strawman_sources} 
                    ${strawman_headers} 
                    $<TARGET_OBJECTS:strawman_vtkm> 
                    ${vtkm_cuda_pipeline}
                    $<TARGET_OBJECTS:strawman_vtkm_renderer>
                    ${vtkm_cuda_renderer})
    else()
        add_library(strawman SHARED ${strawman_sources} ${strawman_headers})
    endif()
else()
    message(STATUS "Building strawman as a static library")
    if(VTKM_FOUND)
        add_library(strawman STATIC 
                    ${strawman_sources} 
                    ${strawman_headers} 
                    $<TARGET_OBJECTS:strawman_vtkm> 
                    ${vtkm_cuda_pipeline}
                    $<TARGET_OBJECTS:strawman_vtkm_renderer> 
                    ${vtkm_cuda_renderer})
    else()
        add_library(strawman STATIC ${strawman_sources} ${strawman_headers})
    endif()
endif()


add_target_compile_flags(TARGET strawman 
                         FLAGS ${VTKm_COMPILE_OPTIONS})


target_link_libraries(strawman ${strawman_thirdparty_libs})

if(CUDA_FOUND)
    target_link_libraries(strawman ${CUDA_CUDART_LIBRARY})
    
endif()


if(TBB_FOUND)
   target_link_libraries(strawman ${TBB_LIBRARIES})
endif()


# install target for serial strawman lib
install(TARGETS strawman
        EXPORT  strawman
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

################################################
# Build Parallel (MPI) version of strawman
################################################
if(MPI_FOUND)
    set(strawman_par_sources utils/strawman_icet_compositor.cpp)
    set(strawman_par_headers utils/strawman_icet_compositor.hpp)
    

    if(VTKM_FOUND)
        # we need to set compiler flags for the MPI case
        # we have to global cxx flags b/c of how the cmake cuda wrappers work 
        # we save / restore the previous set of global flags after we create the cuda 
        # target
        SET(CMAKE_CXX_FLAGS_PREVIOUS ${CMAKE_CXX_FLAGS})
        SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} -D PARALLEL -fPIC" )
        
        #
        #  VTKm library objects
        #
        
        add_library(strawman_vtkm_par OBJECT ${strawman_vtkm_sources} ${strawman_vtkm_headers})
        add_library(strawman_vtkm_renderer_par OBJECT ${strawman_vtkm_renderer_sources} ${strawman_vtkm_renderer_headers})

        if(CUDA_FOUND)
            #
            #  VTKm cuda objects
            #
            cuda_wrap_srcs(strawman_vtkm_pipeline_cuda_par OBJ vtkm_cuda_pipeline_par pipelines/strawman_vtkm_pipeline_backend_cuda.cu)
            cuda_wrap_srcs(strawman_vtkm_renderer_cuda_par OBJ vtkm_cuda_renderer_par pipelines/strawman_vtkm_renderer_cuda.cu)
        endif()
        
        # restore old global cxx flags
        SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_PREVIOUS})
    endif()
   
    if(BUILD_SHARED_LIBS)
        message(STATUS "Building strawman_par as a shared library")
        message(STATUS "${strawman_sources}")
        if(VTKM_FOUND)
            add_library(strawman_par SHARED ${strawman_sources} 
                                            ${strawman_par_sources} 
                                            ${strawman_headers}
                                            ${strawman_par_headers}
                                            $<TARGET_OBJECTS:strawman_vtkm_par>
                                            ${vtkm_cuda_pipeline_par}
                                            $<TARGET_OBJECTS:strawman_vtkm_renderer_par> 
                                            ${vtkm_cuda_renderer_par})
        else()
            add_library(strawman_par SHARED ${strawman_sources} 
                                            ${strawman_par_sources} 
                                            ${strawman_headers}
                                            ${strawman_par_headers} )
        endif()
    else()
        message(STATUS "Building strawman_par as a static library")
        if(VTKM_FOUND)
            add_library(strawman_par STATIC ${strawman_sources} 
                                            ${strawman_par_sources} 
                                            ${strawman_headers}
                                            ${strawman_par_headers}
                                            $<TARGET_OBJECTS:strawman_vtkm_par>
                                            ${vtkm_cuda_pipeline_par}
                                            $<TARGET_OBJECTS:strawman_vtkm_renderer_par> 
                                            ${vtkm_cuda_renderer_par})
        else()
            add_library(strawman_par STATIC ${strawman_sources} 
                                            ${strawman_par_sources} 
                                            ${strawman_headers}
                                            ${strawman_par_headers} )
        endif()
    endif()
    
    add_target_compile_flags(TARGET strawman_par 
                             FLAGS "${MPI_CXX_COMPILE_FLAGS} -D PARALLEL ${VTKm_COMPILE_OPTIONS}")
    #
    # VTKm: We have to add the same compiler flags to both the pipeline and rendering object
    #
    if(VTKM_FOUND)
        add_target_compile_flags(TARGET strawman_vtkm_par 
                                 FLAGS "${MPI_CXX_COMPILE_FLAGS} -D PARALLEL ${VTKm_COMPILE_OPTIONS}")
        add_target_compile_flags(TARGET strawman_vtkm_renderer_par 
                                 FLAGS "${MPI_CXX_COMPILE_FLAGS} -D PARALLEL ${VTKm_COMPILE_OPTIONS}")
    endif()
    
    add_target_link_flags(TARGET strawman_par  
                          FLAGS "${MPI_CXX_LINK_FLAGS}")
    
    target_link_libraries(strawman_par
                          conduit_relay_mpi
                          ${strawman_thirdparty_libs}
                          ${ICET_LIBRARIES}
                          ${MPI_CXX_LIBRARIES})


   
    if(CUDA_FOUND)
        target_link_libraries(strawman_par ${CUDA_CUDART_LIBRARY})
    endif()

    if(TBB_FOUND)
       target_link_libraries(strawman_par ${TBB_LIBRARIES})
    endif()



    # install target for mpi ver
    install(TARGETS strawman_par
            EXPORT  strawman
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )

endif()

########################################
# Extra Install Targets for strawman
########################################

install(FILES ${strawman_headers} DESTINATION include/strawman)


