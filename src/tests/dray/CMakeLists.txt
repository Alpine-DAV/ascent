###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


################################
# Unit Tests
################################

# TODO:

#set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/")
#configure_file(test_config.h.in "${CMAKE_CURRENT_BINARY_DIR}/test_config.h" @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(BASIC_TESTS t_dray_smoke
                t_dray_array
                #t_dray_lines #temporary
                t_dray_balancer
                t_dray_billboard
                t_dray_color_table
                t_dray_font
                t_dray_low_order
                ##t_dray_ambient_occlusion
                ##t_dray_obj_reader
                #t_dray_matrix
                #t_dray_registry
                t_dray_slice
                t_dray_multi_render
                t_dray_dataset_to_node
                t_dray_tri_benchmark
                #t_dray_test
                #t_dray_vec
                #t_dray_high_order_shape
                #t_dray_newton_solve
                #t_dray_cool_shapes
                #t_dray_subdivision
                ##t_dray_performance_tests
                #t_dray_attractors
                ## t_dray_get_sub_bounds  ## has face elements
                t_dray_isosurface
                t_dray_scalar_renderer
                t_dray_volume_render
                t_dray_volume_partials
                t_dray_face_render
                t_dray_reflect
                t_dray_subset
                t_dray_elem_attr
                t_dray_isosurfacing_filter
                t_dray_to_bernstein_filter
                t_dray_external_evals
                t_dray_dsbuilder
                t_dray_lineout
                t_dray_vector_ops
                t_dray_annotations
                #t_dray_sedov
                #t_dray_taylor_green
                ## t_dray_crazy_hex
                #t_dray_crazy_quad
                #t_dray_triangle
                #t_dray_element
                t_dray_mesh_threshold
                t_dray_cell_average
                t_dray_point_average
)

set(MPI_TESTS t_dray_mpi_smoke
              #t_dray_mpi_blueprint_writer
              t_dray_mpi_balance
              t_dray_mpi_faces
              t_dray_mpi_lineout
              t_dray_mpi_scalar_renderer
              t_dray_mpi_redistribute
              t_dray_mpi_volume_render)

set(dray_test_deps dray ascent)

# note: this should come in automatically
#       but not going to temp fate 
if(ENABLE_OPENMP)
  list(APPEND dray_test_deps ${ascent_blt_openmp_deps})
endif()


if(ENABLE_SERIAL)
  message(STATUS "Adding basic dray unit tests")
  foreach(TEST ${BASIC_TESTS})
    message(STATUS "adding test ${TEST}")
    add_cpp_test(TEST ${TEST} DEPENDS_ON ${dray_test_deps})
  endforeach()
endif()

if(MPI_FOUND)
    set(dray_test_mpi_deps dray_mpi ascent)

    message(STATUS "Adding MPI dray unit tests")
    foreach(TEST ${MPI_TESTS})
        message(STATUS "adding MPI test ${TEST}")
        add_cpp_mpi_test(TEST ${TEST} NUM_MPI_TASKS 2 DEPENDS_ON ${dray_test_mpi_deps})
    endforeach()
endif()


# #  add custom command that generates a html img comparison report for our tests
# #  run with "make gen_report"
# #  creates "_output/tout_img_report.html"
# add_custom_target(gen_report
#                   python ${CMAKE_CURRENT_SOURCE_DIR}/gen_img_compare_html_report.py
#                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#                   COMMENT "Generating html summary of image compare tests (tests/dray/_output/tout_img_report.html)")
