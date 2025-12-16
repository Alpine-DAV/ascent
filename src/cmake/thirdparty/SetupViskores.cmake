###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
# Setup Viskores
###############################################################################

if(NOT VISKORES_DIR)
    MESSAGE(FATAL_ERROR "Viskores support needs explicit VISKORES_DIR")
endif()

MESSAGE(STATUS "Looking for Viskores using VISKORES_DIR = ${VISKORES_DIR}")

# use VISKORES_DIR to setup the options that cmake's find Viskores needs
file(GLOB Viskores_DIR "${VISKORES_DIR}/lib/cmake/viskores-*")
if(NOT Viskores_DIR)
    MESSAGE(FATAL_ERROR "Failed to find Viskores at VISKORES_DIR=${VISKORES_DIR}/lib/cmake/vtk-*")
endif()

find_package(Viskores REQUIRED QUIET)

if(ENABLE_CUDA AND NOT Viskores_ENABLE_CUDA)
   message(FATAL_ERROR "VTK-h CUDA support requires Viskores with CUDA support (ENABLE_CUDA == TRUE, however Viskores_ENABLE_CUDA == FALSE")
endif()

set(VISKORES_FOUND TRUE)

set(VISKORES_TARGETS viskores::cont viskores::filter viskores::rendering)
message(STATUS "viskores enalbe mpi:  ${Viskores_ENABLE_MPI}")
message(STATUS "mpi found:  ${MPI_FOUND}")

# add mpi if mfem uses mpi
if(Viskores_ENABLE_MPI)
    if(NOT MPI_FOUND)
        message(FATAL_ERROR "Viskores was built with MPI support (config.mk has VISKORES_MPI_ENABLED = TRUE)"
                             "But ASCENT_MPI_ENABLED = FALSE")
    endif()
    message(STATUS "Viskores was built with MPI support (VISKORES_MPI_ENABLED = TRUE)")
    list(APPEND VISKORES_TARGETS ${ascent_blt_mpi_deps}) 
    set(ASCENT_VISKORES_MPI_ENABLED TRUE)
endif()

if(ENABLE_CUDA)
    # we need to inject the viskores cuda flags into CMAKE_CUDA_FLAGS
    viskores_get_cuda_flags(_fetch_viskores_cuda_flags)
    set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} ${_fetch_viskores_cuda_flags}")
    unset(_fetch_viskores_cuda_flags)
    # we also need
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas --disable-optimizer-constants")
endif()


# VISKORES does not seem to propagate includes it exposes to us, so we have to work
# around this.
file(GLOB VISKORES_LCL_DIR "${VISKORES_DIR}/include/viskores-*/viskores/thirdparty/lcl/viskoreslcl/")
include_directories("${VISKORES_LCL_DIR}")

# VISKORES ridiculous
file(GLOB VISKORES_DIY_DIR "${VISKORES_DIR}/include/viskores-*/viskores/thirdparty/diy/viskoresdiy/include/")
include_directories("${VISKORES_DIY_DIR}")

blt_register_library(NAME viskores
                     LIBRARIES ${VISKORES_TARGETS}
                     )

if(ASCENT_ENABLE_TESTS AND WIN32 AND BUILD_SHARED_LIBS)
    # if we are running tests with dlls, we need path to dlls
    list(APPEND ASCENT_TPL_DLL_PATHS ${VISKORES_DIR}/bin)
endif()