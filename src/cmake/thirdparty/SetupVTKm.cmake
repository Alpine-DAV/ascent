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

#vtkm precision check
#if(NOT VTKm_USE_DOUBLE_PRECISION)
#  message(FATAL_ERROR "Ascent requires VTK-m to be built with double precision (VTKm_USE_64BIT_IDS=OFF and VTKm_USE_DOUBLE_PRECISION=ON)")
#endif()

if(NOT VTKm_USE_DEFAULT_TYPES_FOR_ASCENT)
  message(FATAL_ERROR "Ascent requires VTK-m to be built with default types (VTKm_USE_DEFAULT_TYPES_FOR_ASCENT=ON)")
endif()

if(ENABLE_CUDA AND NOT VTKm_ENABLE_CUDA)
   message(FATAL_ERROR "VTK-h CUDA support requires VTK-m with CUDA support (ENABLE_CUDA == TRUE, however VTKm_ENABLE_CUDA == FALSE")
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
    # we also need
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas --disable-optimizer-constants")
endif()


# VTKM does not seem to propagate includes it exposes to us, so we have to work
# around this.
file(GLOB VTKM_LCL_DIR "${VTKM_DIR}/include/vtkm-*/vtkm/thirdparty/lcl/vtkmlcl/")
include_directories("${VTKM_LCL_DIR}")

# VTKM ridiculous
file(GLOB VTKM_DIY_DIR "${VTKM_DIR}/include/vtkm-*/vtkm/thirdparty/diy/vtkmdiy/include/")
include_directories("${VTKM_DIY_DIR}")

blt_register_library(NAME vtkm
                     LIBRARIES ${VTKM_TARGETS}
                     )

