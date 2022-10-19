# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

# adjust cmake options into proper defs for generate dray_config.h


if(MFEM_FOUND)
    set(DRAY_MFEM_ENABLED TRUE)
    message(STATUS "DRay MFEM Support Enabled")
endif()

if(ENABLE_LOGGING)
    set(DRAY_ENABLE_LOGGING TRUE)
    message(STATUS "DRay Logging enabled")
endif() 

if(DRAY_USE_DOUBLE_PRECISION)
    # this option is used directly in config.h
    message(STATUS "DRay using double precision")
endif()

if(ENABLE_OPENMP)
    set(DRAY_OPENMP_ENABLED TRUE)
    message(STATUS "DRay OpenMP Support Enabled")
endif()

if(DRAY_ENABLE_STATS)
    set(DRAY_STATS TRUE)
    message(STATUS "DRay Stats enabled")
endif()


# create the headr

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/dray_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/dray_config.h")



# devil ray headers

set(dray_headers dray.hpp
                 ${CMAKE_CURRENT_BINARY_DIR}/dray_config.h
                 dray_exports.h
                 array.hpp
                 aabb.hpp
                 additional_color_tables.hpp
                 array_internals.hpp
                 array_internals_base.hpp
                 array_registry.hpp
                 array_utils.hpp
                 color_map.hpp
                 color_table.hpp
                 dray_node_to_dataset.hpp
                 error.hpp
                 error_check.hpp
                 halton.hpp
                 linear_bvh_builder.hpp
                 location.hpp
                 math.hpp
                 integer_utils.hpp
                 morton_codes.hpp
                 matrix.hpp
                 newton_solver.hpp
                 subdivision_search.hpp
                 plane_detector.hpp
                 policies.hpp
                 power_basis.hpp
                 random.hpp
                 range.hpp
                 ray.hpp
                 ray_hit.hpp
                 ref_point.hpp
                 simple_tensor.hpp
                 template_tag.hpp
                 types.hpp
                 transform_3d.hpp
                 triangle_intersection.hpp
                 vec.hpp
                 warning.hpp
                 io/obj_reader.hpp
                 io/blueprint_reader.hpp
                 io/blueprint_low_order.hpp
                 import_order_policy.hpp
                 #
                 rendering/screen_annotator.hpp
                 rendering/billboard.hpp
                 rendering/camera.hpp
                 rendering/colors.hpp
                 rendering/contour.hpp
                 rendering/color_bar_annotator.hpp
                 rendering/device_framebuffer.hpp
                 rendering/font.hpp
                 rendering/font_factory.hpp
                 rendering/fragment.hpp
                 rendering/framebuffer.hpp
                 rendering/low_order_intersectors.hpp
                 rendering/line_renderer.hpp
                 rendering/material.hpp
                 rendering/point_light.hpp
                 rendering/traceable.hpp
                 rendering/renderer.hpp
                 rendering/rasterbuffer.hpp
                 rendering/scalar_buffer.hpp
                 rendering/scalar_renderer.hpp
                 rendering/slice_plane.hpp
                 rendering/surface.hpp
                 rendering/screen_text_annotator.hpp
                 rendering/world_text_annotator.hpp
                 rendering/triangle_mesh.hpp
                 rendering/volume.hpp
                 rendering/volume_partial.hpp
                 rendering/volume_shader.hpp
                 rendering/world_annotator.hpp

                 queries/lineout.hpp
                 queries/point_location.hpp

                 utils/color_buffer_utils.hpp
                 utils/data_logger.hpp
                 utils/png_encoder.hpp
                 utils/png_decoder.hpp
                 utils/png_compare.hpp
                 utils/ray_utils.hpp
                 utils/string_utils.hpp
                 utils/mpi_utils.hpp
                 utils/timer.hpp
                 utils/yaml_writer.hpp

                 ambient_occlusion.hpp
                 intersection_context.hpp
                 binomial.hpp
                 constants.hpp
                 utils/stats.hpp
                 utils/appstats.hpp
                 utils/dataset_builder.hpp

                 dispatcher.hpp

                 data_model/collection.hpp
                 data_model/data_set.hpp
                 data_model/bernstein_basis.hpp
                 data_model/bezier_simplex.hpp
                 data_model/subref.hpp
                 data_model/elem_attr.hpp
                 data_model/ref_space.hpp
                 data_model/element.hpp
                 data_model/detached_element.hpp
                 data_model/elem_utils.hpp
                 data_model/elem_ops.hpp
                 data_model/iso_ops.hpp
                 data_model/pos_tensor_element.tcc
                 data_model/pos_simplex_element.tcc
                 data_model/subpatch.hpp
                 data_model/mesh_utils.hpp
                 data_model/field.hpp
                 data_model/unstructured_field.hpp
                 data_model/grid_function.hpp
                 data_model/unstructured_mesh.hpp
                 data_model/mesh.hpp

                 filters/mesh_boundary.hpp
                 filters/reflect.hpp
                 filters/redistribute.hpp
                 filters/subset.hpp
                 filters/volume_balance.hpp
                 filters/isosurfacing.hpp
                 filters/to_bernstein.hpp
                 filters/vector_component.hpp

                 #filters/attractor_map.hpp
                 filters/surface_triangle.hpp

                 synthetic/affine_radial.hpp
                 synthetic/spiral_sample.hpp
                 synthetic/tet_sphere_sample.hpp

)

# devil ray sources

set(dray_sources dray.cpp
                 array.cpp
                 array_internals.cpp
                 array_internals_base.cpp
                 array_registry.cpp
                 color_map.cpp
                 color_table.cpp
                 dray_node_to_dataset.cpp
                 linear_bvh_builder.cpp
                 location.cpp
                 matrix.cpp
                 subdivision_search.cpp
                 ray.cpp
                 ray_hit.cpp
                 warning.cpp
                 vec.cpp
                 utils/color_buffer_utils.cpp
                 utils/data_logger.cpp
                 utils/png_encoder.cpp
                 utils/png_decoder.cpp
                 utils/png_compare.cpp
                 utils/ray_utils.cpp
                 utils/string_utils.cpp
                 utils/mpi_utils.cpp

                 queries/lineout.cpp
                 queries/point_location.cpp

                 ambient_occlusion.cpp
                 intersection_context.cpp
                 binomial.cpp

                 io/obj_reader.cpp
                 io/blueprint_reader.cpp
                 io/blueprint_low_order.cpp

                 rendering/screen_annotator.cpp
                 rendering/billboard.cpp
                 rendering/camera.cpp
                 rendering/color_bar_annotator.cpp
                 rendering/contour.cpp
                 rendering/font.cpp
                 rendering/font_factory.cpp
                 rendering/fragment.cpp
                 rendering/framebuffer.cpp
                 rendering/line_renderer.cpp
                 rendering/traceable.cpp
                 rendering/material.cpp
                 rendering/point_light.cpp
                 rendering/renderer.cpp
                 rendering/scalar_buffer.cpp
                 rendering/scalar_renderer.cpp
                 rendering/slice_plane.cpp
                 rendering/surface.cpp
                 rendering/texture2d.cpp
                 rendering/screen_text_annotator.cpp
                 rendering/world_text_annotator.cpp
                 rendering/triangle_mesh.cpp
                 rendering/volume.cpp
                 rendering/volume_partial.cpp
                 rendering/world_annotator.cpp
                 # utils
                 utils/appstats.cpp
                 utils/dataset_builder.cpp
                 dispatcher.cpp
                 # data model
                 data_model/collection.cpp
                 data_model/data_set.cpp
                 data_model/element.cpp
                 data_model/pos_tensor_element.cpp
                 data_model/pos_simplex_element.cpp
                 data_model/subref.cpp
                 data_model/iso_ops.cpp
                 data_model/grid_function.cpp
                 data_model/unstructured_mesh.cpp
                 data_model/mesh_utils.cpp
                 data_model/unstructured_field.cpp
                 # filters
                 filters/mesh_boundary.cpp
                 filters/reflect.cpp
                 filters/redistribute.cpp
                 filters/subset.cpp
                 filters/volume_balance.cpp
                 filters/isosurfacing.cpp
                 filters/to_bernstein.cpp
                 filters/vector_component.cpp
                 #filters/attractor_map.cpp
                 filters/surface_triangle.cpp
                 # synthetic
                 synthetic/affine_radial.cpp
                 synthetic/spiral_sample.cpp
                 synthetic/tet_sphere_sample.cpp
)


################################################
# third party required deps
################################################
set(dray_thirdparty_libs
    RAJA
    umpire
    conduit::conduit
    ascent_lodepng)

if(MFEM_FOUND)
    list(APPEND dray_thirdparty_libs mfem)
    list(APPEND dray_headers io/mfem_reader.hpp
                             utils/mfem_utils.hpp
                             mfem2dray.hpp)

    list(APPEND dray_sources io/mfem_reader.cpp
                             utils/mfem_utils.cpp
                             mfem2dray.cpp)
endif()

################################################
# openmp support
################################################
if(ENABLE_OPENMP)
    list(APPEND dray_thirdparty_libs ${ascent_blt_openmp_deps})
endif()

################################################
# cuda support
################################################
if(ENABLE_CUDA)
    list(APPEND dray_thirdparty_libs cuda)
    # Building the BVH requires us to propagate bounding boxes in a tree
    # from the bottom up. This means one thread can update aabbs in a node
    # before terminating and letting another thread take over processing
    # of the sub-tree. That said, there are no promises about which thread,
    # and apparently there is no cache coherence between L1 caches so the
    # other thread can might not get the updated data, destroying entire
    # subtrees of the bvh. This doesn't happen vary often, but does happen.
    # This compile flags disables L1 cache for this source file, preventing
    # the issue. This line is the culmination of 10 hours of debugging,
    # during which I questioned my own reality (ML). Slight optimization
    # would be to move the one function to another file.
    set_source_files_properties(linear_bvh_builder.cpp PROPERTIES COMPILE_FLAGS "-Xptxas -dlcm=cg")
endif()


################################################
# hip support
################################################
if(ENABLE_HIP)
    list(APPEND dray_thirdparty_libs blt_hip blt_hip_runtime)
endif()


#################################e#######
# Serial Devil Ray (i.e., no mpi)
########################################
if(ENABLE_SERIAL)
    blt_add_library(NAME dray
                    SOURCES ${dray_sources}
                    HEADERS ${dray_headers}
                    DEPENDS_ON ${dray_thirdparty_libs} apcomp
    )

    # build time only defs
    target_compile_definitions(dray PRIVATE ASCENT_EXPORTS_FLAG)

    if(ENABLE_HIDDEN_VISIBILITY)
        set_target_properties(dray PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

    # build includes
    # this allows us to include as <dray/header.hpp>
    target_include_directories(dray PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(dray PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(dray PUBLIC $<INSTALL_INTERFACE:include/ascent/dray>)

    install(TARGETS dray
            EXPORT ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )
endif()

if(MPI_FOUND)
    ########################################
    # MPI Devil Ray
    ########################################
    set(dray_mpi_deps ${dray_thirdparty_libs}
                      ${ascent_blt_mpi_deps}
                      conduit::conduit_mpi apcomp_mpi)

    blt_add_library(NAME dray_mpi
                    SOURCES ${dray_sources}
                    HEADERS ${dray_headers}
                    DEPENDS_ON ${dray_mpi_deps})

    # build time only defs
    target_compile_definitions(dray_mpi PRIVATE ASCENT_EXPORTS_FLAG)
    target_compile_definitions(dray_mpi PRIVATE DRAY_MPI_ENABLED)

    if(ENABLE_HIDDEN_VISIBILITY)
        set_target_properties(dray_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()
  
    # build includes
    # this allows us to include as <dray/header.hpp>
    target_include_directories(dray_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/>)
    target_include_directories(dray_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/libs/>)

    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(dray_mpi PUBLIC $<INSTALL_INTERFACE:include/ascent/dray>)

    install(TARGETS dray_mpi
            EXPORT ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib)
endif()


######################################################
# Install headers while preserve directory structure
######################################################
macro(install_headers_with_directory header_list)
  foreach(header ${${header_list}})
    string(REGEX MATCH "(.*)[/\\]" DIR ${header})
    install(FILES ${header} DESTINATION include/ascent/dray/${DIR})
  endforeach(header)
endmacro(install_headers_with_directory)

install_headers_with_directory(dray_headers)


if(ENABLE_UTILS)
   message(status "Dray Utilities Enabled")
   add_subdirectory(utilities)
endif()


