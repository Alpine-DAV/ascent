###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# Setup VTKm
###############################################################################

if(NOT VTKM_DIR)
    MESSAGE(FATAL_ERROR "VTKm support needs explicit VTKM_DIR")
endif()

MESSAGE(STATUS "Looking for VTKm using VTKM_DIR = ${VTKM_DIR}")

# use VTKM_DIR to setup the options that cmake's find VTKm needs
file(GLOB VTKm_DIR "${VTKM_DIR}/lib/cmake/vtkm-*")
if(NOT VTKm_DIR)
    MESSAGE(FATAL_ERROR "Failed to find VTKm at VTKM_DIR=${VTKM_DIR}/lib/cmake/vtk-*")
endif()

find_package(VTKm REQUIRED QUIET)

if(ENABLE_CUDA AND NOT VTKm_ENABLE_CUDA)
   message(FATAL_ERROR "Ascent CUDA support requires VTK-m with CUDA support (ENABLE_CUDA == TRUE, however VTKm_ENABLE_CUDA == FALSE")
endif()

if(ENABLE_CUDA AND BUILD_SHARED_LIBS)
  if(VTKm_VERSION VERSION_LESS "1.7.0")
    message(FATAL_ERROR "Cannot build shared libs with CUDA when VTKm is < v1.7.0")
  endif()
endif()

set(VTKM_FOUND TRUE)

set(VTKM_TARGETS vtkm_cont vtkm_filter vtkm_rendering)

if(ENABLE_CUDA)
    # we need to inject the vtkm cuda flags into CMAKE_CUDA_FLAGS
    vtkm_get_cuda_flags(_fetch_vtkm_cuda_flags)
    set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} ${_fetch_vtkm_cuda_flags}")
    unset(_fetch_vtkm_cuda_flags)
endif()


blt_register_library(NAME vtkm
                     LIBRARIES ${VTKM_TARGETS}
                     )
