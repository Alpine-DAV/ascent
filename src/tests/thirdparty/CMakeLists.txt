###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

################################
# Basic TPL Tests
################################

IF(VTKH_FOUND AND ENABLE_SERIAL)
    message(STATUS "Adding vtk-h lib unit tests")
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
    message(STATUS "Adding mfem lib unit tests")
    set(mfem_smoke_deps mfem conduit)
    # mfem does not have both a serial version and an mpi
    # version. So if we compiled with MPI on, the we need
    # to set the correct dependency
    if(MPI_FOUND AND ENABLE_MPI)
        list(APPEND mfem_smoke_deps mpi conduit)
    endif()
    add_cpp_test(TEST t_mfem_smoke DEPENDS_ON ${mfem_smoke_deps})
endif()




