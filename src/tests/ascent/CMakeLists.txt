###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


################################
# Unit Tests
################################

################################
# Core Ascent Unit Tests
################################
set(BASIC_TESTS t_ascent_smoke
                t_ascent_runtime_options
                t_ascent_data_binning
                t_ascent_utils
                t_ascent_memory
                t_ascent_derived
                t_ascent_empty_runtime
                t_ascent_expressions
                t_ascent_cinema_a
                t_ascent_render_2d
                t_ascent_render_3d
                t_ascent_render_2d_poly
                t_ascent_render_3d_poly
                t_ascent_web_flow
                t_ascent_web_main
                t_ascent_clip
                t_ascent_scalar_renderer
                t_ascent_multi_topo
                t_ascent_gradient
                t_ascent_qcriterion
                t_ascent_divergence
                t_ascent_vorticity
                t_ascent_relay
                t_ascent_htg
                t_ascent_flatten
                t_ascent_partition
                t_ascent_clip_with_field
                t_ascent_contour
                t_ascent_iso_volume
                t_ascent_image_compare
                t_ascent_pipelines_to_pipelines
                t_ascent_threshold
                t_ascent_slice
                t_ascent_scale
                t_ascent_particle_advection
                t_ascent_patch_amr
                t_ascent_vector_ops
                t_ascent_flow_runtime
                t_ascent_recenter
                t_ascent_rover
                t_ascent_lagrangian
                t_ascent_log
                t_ascent_amr
                t_ascent_queries
                t_ascent_failed_pipeline
                t_ascent_triggers
                t_ascent_blueprint_reductions
                t_ascent_sampling
                t_ascent_hola)

set(MPI_TESTS  t_ascent_mpi_smoke
               t_ascent_mpi_empty_runtime
               t_ascent_mpi_derived
               t_ascent_mpi_expressions
	       t_ascent_mpi_flatten
	       t_ascent_mpi_partition
               t_ascent_mpi_render_2d
               t_ascent_mpi_render_3d
               t_ascent_mpi_statistics
               t_ascent_mpi_multi_topo
               t_ascent_mpi_slice)

# t_ascent_hola_mpi uses 8 mpi tasks, so its added manually
# same for t_ascent_babelflow_pmt_mpi and t_ascent_babelflow_comp_mpi

# include the "ascent" pipeline
if(VTKM_FOUND)
   list(APPEND BASIC_TESTS t_ascent_ascent_runtime)
   list(APPEND VTKH_DEP_TESTS t_ascent_vtkh_data_adapter)
   list(APPEND MPI_TESTS   t_ascent_mpi_ascent_runtime
                           t_ascent_mpi_relay_extract)
endif()


# adios tests
if(FIDES_FOUND)
   list(APPEND MPI_TESTS t_ascent_mpi_adios_extract)
endif()

if(PYTHON_FOUND AND ENABLE_PYTHON)
    list(APPEND BASIC_TESTS t_ascent_python_extract)
    list(APPEND MPI_TESTS t_ascent_mpi_python_extract)
endif()

if(DRAY_FOUND)
    list(APPEND BASIC_TESTS t_ascent_dray)
endif()

if(GENTEN_FOUND)
    list(APPEND BASIC_TESTS t_ascent_genten_cokurt)
endif()


################################
# Add main tests
################################
if (ENABLE_SERIAL)
    message(STATUS "Adding ascent lib unit tests")
    foreach(TEST ${BASIC_TESTS})
        add_cpp_test(TEST ${TEST} DEPENDS_ON ascent)
    endforeach()

    foreach(TEST ${VTKH_DEP_TESTS})
        add_cpp_test(TEST ${TEST} DEPENDS_ON ascent)
    endforeach()

    if(VTKM_FOUND)
      # special test case where we test vtkm conversions
       vtkm_add_target_information(t_ascent_vtkh_data_adapter
                                   DEVICE_SOURCES t_ascent_vtkh_data_adapter.cpp)
       set_target_properties(t_ascent_vtkh_data_adapter PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()
endif()
################################
# Add optional tests
################################

if(MPI_FOUND AND ENABLE_MPI)
    message(STATUS "MPI enabled: Adding related unit tests")
    foreach(TEST ${MPI_TESTS})
        # this uses 2 procs
        add_cpp_mpi_test(TEST ${TEST} NUM_MPI_TASKS 2 DEPENDS_ON ascent_mpi)
    endforeach()

    # add the hola mpi test which uses 8 ranks
    add_cpp_mpi_test(TEST t_ascent_hola_mpi NUM_MPI_TASKS 8 DEPENDS_ON ascent_mpi)

  if(BABELFLOW_FOUND)
    # add the babelflow pmt mpi test
    add_cpp_mpi_test(TEST t_ascent_babelflow_pmt_mpi NUM_MPI_TASKS 8 DEPENDS_ON ascent_mpi)
    if(DRAY_FOUND)
      # babelflow compositing test requires dray
      add_cpp_mpi_test(TEST t_ascent_babelflow_comp_mpi NUM_MPI_TASKS 8 DEPENDS_ON ascent_mpi)
    endif()
  endif()
else()
    message(STATUS "MPI disabled: Skipping related tests")
endif()

if(PYTHON_FOUND AND ENABLE_PYTHON)
    add_subdirectory("python")
else()
    message(STATUS "Python disabled: Skipping ascent python module tests")
endif()


if(FORTRAN_FOUND AND ENABLE_FORTRAN)
     add_subdirectory("fortran")
else()
     message(STATUS "Fortran disabled: Skipping ascent fortran interface tests")
endif()
