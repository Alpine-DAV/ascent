###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

################################
# Unit Tests
################################


################################
# Core VTK-h Unit Tests
################################
set(BASIC_TESTS t_vtk-h_smoke
                t_vtk-h_dataset
                t_vtk-h_clip
                t_vtk-h_clip_field
                t_vtk-h_vector_ops
                t_vtk-h_device_control
                t_vtk-h_empty_data
                t_vtk-h_gradient
                t_vtk-h_ghost_stripper
                t_vtk-h_iso_volume
                t_vtk-h_no_op
                t_vtk-h_marching_cubes
                t_vtk-h_lagrangian
                t_vtk-h_log
                t_vtk-h_threshold
                t_vtk-h_point_transform
                t_vtk-h_mesh_renderer
                t_vtk-h_mesh_quality
                t_vtk-h_multi_render
                t_vtk-h_point_renderer
                t_vtk-h_raytracer
                t_vtk-h_render
                t_vtk-h_slice
                t_vtk-h_volume_renderer
                )

# Contour tree depends on the VTK-m build type.
# If vtkm was built with/out mpi support, we only run the
# non-mpi contour tree test
if(NOT VTKm_ENABLE_MPI AND VTKH_ENABLE_FILTER_CONTOUR_TREE)
    list(APPEND BASIC_TESTS t_vtk-h_contour_tree)
endif()

set(CUDA_TESTS t_vtk-h_cuda)

set(MPI_TESTS t_vtk-h_smoke_par
              t_vtk-h_dataset_par
              t_vtk-h_no_op_par
              t_vtk-h_histogram_par
              t_vtk-h_statistics_par
              t_vtk-h_marching_cubes_par
              t_vtk-h_multi_render_par
              t_vtk-h_particle_advection_par
              t_vtk-h_scalar_renderer_par
              t_vtk-h_raytracer_par
              t_vtk-h_sampling_par
              t_vtk-h_volume_renderer_par
              )

# Contour tree depends on the VTK-m build type.
# If vtkm was built with mpi support, we only run the
# par contour tree test
if(VTKm_ENABLE_MPI AND VTKH_ENABLE_FILTER_CONTOUR_TREE)
    list(APPEND MPI_TESTS t_vtk-h_contour_tree_par)
endif()


################################
# Add main tests
################################
if (ENABLE_SERIAL)
    message(STATUS "Adding vtk-h lib unit tests")
    foreach(TEST ${BASIC_TESTS})
        if(ENABLE_CUDA)
            add_cpp_test(TEST ${TEST} DEPENDS_ON vtkh cuda)
            vtkm_add_target_information(${TEST} DEVICE_SOURCES ${TEST}.cpp)
            set_target_properties(${TEST} PROPERTIES LINKER_LANGUAGE CUDA)
        else()
            add_cpp_test(TEST ${TEST} DEPENDS_ON vtkh)
            set_target_properties(${TEST} PROPERTIES LINKER_LANGUAGE CXX)
        endif()
        set_target_properties(${TEST} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endforeach()
    if(CUDA_FOUND)
        foreach(TEST ${CUDA_TESTS})
              add_cpp_test(TEST ${TEST} DEPENDS_ON vtkh cuda)
        endforeach()
    endif()
endif()

################################
# Add optional tests
################################
if(MPI_FOUND)
    message(STATUS "MPI enabled: Adding vtk-h mpi unit tests")
    foreach(TEST ${MPI_TESTS})
        # this uses 2 procs
        if(ENABLE_CUDA)
            add_cpp_mpi_test(TEST ${TEST} NUM_PROCS 2 DEPENDS_ON vtkh_mpi cuda)
            vtkm_add_target_information(${TEST} DEVICE_SOURCES ${TEST}.cpp)
        else()
            add_cpp_mpi_test(TEST ${TEST} NUM_PROCS 2 DEPENDS_ON vtkh_mpi)
        endif()
        set_target_properties(${TEST} PROPERTIES CXX_VISIBILITY_PRESET hidden)
        target_compile_definitions(${TEST} PRIVATE VTKH_PARALLEL)
    endforeach()
else()
    message(STATUS "MPI disabled: Skipping vtk-h mpi unit tests")
endif()
