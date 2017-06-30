###############################################################################
#
# file: src/vtk-h/CMakeLists.txt
#
###############################################################################


set(VTKH_VTKM_ENABLED ${VTKM_FOUND})

if(VTKH_VTKM_ENABLED)
    set(VTKH_VTKM_USE_CUDA ${CUDA_FOUND})
    set(VTKH_VTKM_USE_TBB  ${TBB_FOUND})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


set(VTKH_USE_OPENMP  ${OPENMP_FOUND})

#configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/vtkh_config.h.in"
#                "${CMAKE_CURRENT_BINARY_DIR}/vtkh_config.h")

set(vtkh_sources
    # vtkh interface
    vtkh.cpp
    vtkh_data_set.cpp
    utils/vtkh_png_encoder.cpp
    )

set(vtkh_headers
    #${CMAKE_CURRENT_BINARY_DIR}/vtkh_config.h
    vtkh.hpp
    vtkh_data_set.hpp
    vtkh_error.hpp
    vtkh_filter.hpp
    utils/vtkm_dataset_info.hpp
    utils/vtkh_png_encoder.hpp
    )

##############################
# Add additional sources
#############################

set(vtkh_rendering_headers
    rendering/compositing/vtkh_compositor.hpp
    rendering/vtkh_image.hpp
    rendering/vtkh_renderer.hpp
    rendering/vtkh_renderer_ray_tracer.hpp
    rendering/vtkh_renderer_volume.hpp
    ) 

set(vtkh_rendering_sources
    rendering/compositing/vtkh_compositor.cpp
    rendering/vtkh_renderer.cpp
    rendering/vtkh_renderer_ray_tracer.cpp
    rendering/vtkh_renderer_volume.cpp
    )
list (APPEND vtkh_sources ${vtkh_rendering_sources})
list (APPEND vtkh_headers ${vtkh_rendering_headers})

##############################
# Add lodepng 
#############################
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty_builtin/lodepng/ ${CMAKE_CURRENT_SOURCE_DIR}/utils/lodepng)

##################

set(vtkh_thirdparty_libs )

if(VTKM_FOUND)
    list(APPEND vtkh_thirdparty_libs vtkm)
    list(APPEND vtkh_thirdparty_libs vtkm_cont)
    list(APPEND vtkh_thirdparty_libs vtkm_rendering)
endif()

##########################################
# Build a serial version of vtkh
##########################################

if(BUILD_SHARED_LIBS)
    message(STATUS "Building vtk-h as a shared library")
     add_library(vtkh SHARED 
                 ${vtkh_sources} 
                 $<TARGET_OBJECTS:lodepng>
                 ${vtkh_headers})
else()
    message(STATUS "Building vtk-h as a static library")
    add_library(vtkh STATIC 
                ${vtkh_sources} 
                 $<TARGET_OBJECTS:lodepng>
                ${vtkh_headers})
endif()

##########################################
# Ensure we use c++11
##########################################

set_target_properties(vtkh PROPERTIES
                      CXX_STANDARD 11
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENTIONS NO)

add_target_compile_flags(TARGET vtkh 
                         FLAGS ${VTKm_COMPILE_OPTIONS})


target_link_libraries(vtkh ${vtkh_thirdparty_libs})

if(CUDA_FOUND)
    target_link_libraries(vtkh ${CUDA_CUDART_LIBRARY})
endif()


if(TBB_FOUND)
   target_link_libraries(vtkh ${TBB_LIBRARIES})
endif()


# install target for serial vtkh lib
install(TARGETS vtkh 
        EXPORT  vtkh 
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

################################################
# Build Parallel (MPI) version of vtkh
################################################
if(MPI_FOUND)
    set(vtkh_par_sources )
                         
    set(vtkh_par_headers )
      
    set(vtkh_par_rendering_headers
        rendering/compositing/vtkh_diy_collect.hpp
        rendering/compositing/vtkh_diy_compositor.hpp
        rendering/compositing/vtkh_diy_direct_send.hpp
        rendering/compositing/vtkh_diy_image_block.hpp
        rendering/compositing/vtkh_diy_radix_k.hpp
        ) 

    set(vtkh_par_rendering_sources
        rendering/vtkh_renderer.cpp
        rendering/vtkh_renderer_ray_tracer.cpp
        rendering/compositing/vtkh_diy_compositor.cpp
        rendering/compositing/vtkh_diy_direct_send.cpp
        rendering/compositing/vtkh_diy_radix_k.cpp
        )
    list (APPEND vtkh_par_sources ${vtkh_par_rendering_sources})
    list (APPEND vtkh_par_headers ${vtkh_par_rendering_headers})


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
        message(STATUS "Building vtkh_par as a shared library")
        add_library(vtkh_par SHARED ${vtkh_sources} 
                                     ${vtkh_par_sources} 
                                     ${vtkh_headers}
                                     $<TARGET_OBJECTS:lodepng>
                                     ${vtkh_par_headers})
    else()
        message(STATUS "Building vtkh_par as a static library")
        add_library(vtkh_par STATIC ${vtkh_sources} 
                                     ${vtkh_par_sources} 
                                     ${vtkh_headers}
                                     $<TARGET_OBJECTS:lodepng>
                                     ${vtkh_par_headers})
    endif()
    
    ##########################################
    # Ensure we use c++11
    ##########################################

    set_target_properties(vtkh_par PROPERTIES
                          CXX_STANDARD 11
                          CXX_STANDARD_REQUIRED YES
                          CXX_EXTENTIONS NO)

    add_target_compile_flags(TARGET vtkh_par 
                             FLAGS "${MPI_CXX_COMPILE_FLAGS} -D PARALLEL ${VTKm_COMPILE_OPTIONS}")
        
    add_target_link_flags(TARGET vtkh_par  
                          FLAGS "${MPI_CXX_LINK_FLAGS}")
    
    target_link_libraries(vtkh_par
                          conduit_relay_mpi
                          ${vtkh_thirdparty_libs}
                          ${ICET_LIBRARIES}
                          ${MPI_CXX_LIBRARIES})


   
    if(CUDA_FOUND)
        target_link_libraries(vtkh_par ${CUDA_CUDART_LIBRARY})
    endif()

    if(TBB_FOUND)
       target_link_libraries(vtkh_par ${TBB_LIBRARIES})
    endif()
  
    # install target for mpi ver
    install(TARGETS vtkh_par
            EXPORT  vtkh
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )

endif()

########################################
# Extra Install Targets for vtkh
########################################

install(FILES ${vtkh_headers} DESTINATION include/vtkh)


