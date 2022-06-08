###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

################################
# TPL Smoke Tests
################################

IF(VTKM_FOUND AND ENABLE_SERIAL)
    message(STATUS "Adding VTK-m smoke test")
    set(vtkm_smoke_deps vtkm)
    
    if(CUDA_FOUND)
        list(APPEND vtkm_smoke_deps cuda)
    endif()

    add_cpp_test(TEST t_vtkm_smoke DEPENDS_ON ${vtkm_smoke_deps})
    vtkm_add_target_information(t_vtkm_smoke
                                DEVICE_SOURCES t_vtkm_smoke.cpp)
    set_target_properties(t_vtkm_smoke PROPERTIES CXX_VISIBILITY_PRESET hidden)

    target_compile_definitions(t_vtkm_smoke PRIVATE "VTKH_FORCE_SERIAL")

    if (ENABLE_OPENMP)
        target_compile_definitions(t_vtkm_smoke PRIVATE "VTKH_FORCE_OPENMP")
    endif()

    if (CUDA_FOUND)
        target_compile_definitions(t_vtkm_smoke PRIVATE "VTKH_FORCE_CUDA")
    endif()

endif()

IF(VTKH_FOUND AND ENABLE_SERIAL)
    message(STATUS "Adding vtk-h lib smoke test")
    # also add vtkm, i suspect compile flags aren't transitive
    # for some reason, and this can undermine cuda builds
    set(vtkh_smoke_deps vtkm vtkh)
    if(CUDA_FOUND)
        list(APPEND vtkh_smoke_deps cuda)
    endif()
    add_cpp_test(TEST t_vtkh_smoke DEPENDS_ON ${vtkh_smoke_deps})
    vtkm_add_target_information(t_vtkh_smoke
                                DEVICE_SOURCES t_vtkh_smoke.cpp)
    set_target_properties(t_vtkh_smoke PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()


IF(MFEM_FOUND)
    message(STATUS "Adding mfem smoke test")
    set(mfem_smoke_deps mfem conduit)
    # mfem does not have both a serial version and an mpi
    # version. So if we compiled with MPI on, the we need
    # to set the correct dependency
    if(MPI_FOUND AND ENABLE_MPI)
        list(APPEND mfem_smoke_deps mpi conduit)
    endif()
    add_cpp_test(TEST t_mfem_smoke DEPENDS_ON ${mfem_smoke_deps})
endif()

