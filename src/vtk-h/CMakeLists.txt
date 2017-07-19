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

set(VTKH_USE_OPENMP  ${OPENMP_FOUND})

set(vtkh_sources
    vtkh.cpp
    vtkh_data_set.cpp
    utils/vtkh_png_encoder.cpp
    )

set(vtkh_headers
    vtkh.hpp
    vtkh_data_set.hpp
    vtkh_error.hpp
    vtkh_filter.hpp
    utils/vtkm_dataset_info.hpp
    utils/vtkh_png_encoder.hpp
    )

##############################
# Add rendering sources
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
# Add filters sources
#############################

set(vtkh_filter_headers
     vtkh_marching_cubes.hpp
     vtkh_clip.hpp
   )

set(vtkh_filter_sources
     vtkh_marching_cubes.cpp
     vtkh_clip.cpp
   )

list (APPEND vtkh_sources ${vtkh_filter_sources})
list (APPEND vtkh_headers ${vtkh_filter_headers})


##################

set(vtkh_thirdparty_libs )

if(VTKM_FOUND)
    list(APPEND vtkh_thirdparty_libs tbb)
endif()

if(VTKM_FOUND)
    list(APPEND vtkh_thirdparty_libs vtkm)
endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

##########################################
# Build a serial version of vtkh
##########################################

blt_add_library(
    NAME        vtkh
    SOURCES     ${vtkh_sources} $<TARGET_OBJECTS:lodepng>
    HEADERS     ${vtkh_headers}
    DEPENDS_ON  ${vtkh_thirdparty_libs})

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
    set(vtkh_par_sources
        rendering/compositing/vtkh_diy_collect.hpp
        rendering/compositing/vtkh_diy_compositor.hpp
        rendering/compositing/vtkh_diy_direct_send.hpp
        rendering/compositing/vtkh_diy_image_block.hpp
        rendering/compositing/vtkh_diy_radix_k.hpp
        ) 

    set(vtkh_par_headers
        rendering/vtkh_renderer.cpp
        rendering/vtkh_renderer_ray_tracer.cpp
        rendering/compositing/vtkh_diy_compositor.cpp
        rendering/compositing/vtkh_diy_direct_send.cpp
        rendering/compositing/vtkh_diy_radix_k.cpp
        )

    blt_add_library(
        NAME        vtkh_par
        SOURCES     ${vtkh_sources} ${vtkh_par_sources} $<TARGET_OBJECTS:lodepng>
        HEADERS     ${vtkh_par_headers}
        DEPENDS_ON  ${vtkh_thirdparty_libs} mpi icet)
  
    blt_add_target_compile_flags(TO vtkh_par FLAGS "-D PARALLEL")
  
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


