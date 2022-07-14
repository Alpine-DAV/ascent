###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

include(CMakeFindDependencyMacro)

###############################################################################
# Setup OpenMP
###############################################################################
if(ASCENT_OPENMP_ENABLED)
    # config openmp if not already found
    if(NOT TARGET OpenMP::OpenMP_CXX)
        find_dependency(OpenMP REQUIRED)
    endif()
endif()

###############################################################################
# Setup Conduit
###############################################################################
# If ZZZ_DIR not set, use known install path for Conduit and VTK-h
if(NOT CONDUIT_DIR)
    set(CONDUIT_DIR ${ASCENT_CONDUIT_DIR})
endif()

###############################################################################
# Check for CONDUIT_DIR
###############################################################################
if(NOT CONDUIT_DIR)
    message(FATAL_ERROR "Could not find Conduit. Conduit requires explicit CONDUIT_DIR.")
endif()

if(NOT EXISTS ${CONDUIT_DIR}/lib/cmake/conduit/conduit.cmake)
    message(FATAL_ERROR "Could not find Conduit CMake include file (${CONDUIT_DIR}/lib/cmake/conduit/conduit.cmake)")
endif()

###############################################################################
# Import Conduit's CMake targets
###############################################################################
find_dependency(Conduit REQUIRED
                NO_DEFAULT_PATH
                PATHS ${CONDUIT_DIR}/lib/cmake)

###############################################################################
# Setup VTK-h (external)
###############################################################################
if(NOT VTKH_DIR)
    set(VTKH_DIR ${ASCENT_VTKH_DIR})
endif()

if(VTKH_DIR)
    if(NOT EXISTS ${VTKH_DIR}/lib/VTKhConfig.cmake)
        message(FATAL_ERROR "Could not find VTKh CMake include file (${VTKH_DIR}/lib/VTKhConfig.cmake)")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(VTKh REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${VTKH_DIR}/lib/)
endif()

###############################################################################
# Setup VTK-m
###############################################################################
if(NOT VTKM_DIR)
    set(VTKM_DIR ${ASCENT_VTKM_DIR})
endif()

if(VTKM_DIR)
    # use VTKM_DIR to setup the options that cmake's find VTKm needs
    file(GLOB VTKm_DIR "${VTKM_DIR}/lib/cmake/vtkm-*")
    if(NOT VTKm_DIR)
        message(FATAL_ERROR "Failed to find VTKm at VTKM_DIR=${VTKM_DIR}/lib/cmake/vtk-*")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(VTKm REQUIRED)
endif()

###############################################################################
# Setup Devil Ray
###############################################################################
if(NOT DRAY_DIR)
    set(DRAY_DIR ${ASCENT_DRAY_DIR})
endif()

if(DRAY_DIR)
    if(NOT EXISTS ${DRAY_DIR}/lib/cmake/DRayConfig.cmake)
        message(FATAL_ERROR "Could not find Devil Ray CMake include file (${DRAY_DIR}/lib/cmake/DRayConfig.cmake)")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(DRay REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${DRAY_DIR}/lib/cmake/)
endif()

###############################################################################
# Setup Umpire
###############################################################################
if(NOT UMPIRE_DIR)
    set(UMPIRE_DIR ${ASCENT_UMPIRE_DIR})
endif()

if(UMPIRE_DIR)
    set(_UMPIRE_SEARCH_PATH)
    if(EXISTS ${UMPIRE_DIR}/share/umpire/cmake)
      # old install layout
      set(_UMPIRE_SEARCH_PATH ${UMPIRE_DIR}/share/umpire/cmake)
    else()
      # new install layout
      set(_UMPIRE_SEARCH_PATH ${UMPIRE_DIR}/lib/cmake/umpire)
    endif()
    
    if(NOT EXISTS ${_UMPIRE_SEARCH_PATH})
        message(FATAL_ERROR "Could not find Umpire CMake include file (${_UMPIRE_SEARCH_PATH})")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(umpire REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${_UMPIRE_SEARCH_PATH})
endif()

###############################################################################
# Setup Camp
###############################################################################
if(NOT CAMP_DIR)
    set(CAMP_DIR ${ASCENT_CAMP_DIR})
endif()

if(CAMP_DIR)
    set(_CAMP_SEARCH_PATH)
    if(EXISTS ${CAMP_DIR}/share/camp/cmake)
      # old install layout ?
      set(_CAMP_SEARCH_PATH ${CAMP_DIR}/share/camp/cmake)
    else()
      # new install layout ?
      set(_CAMP_SEARCH_PATH ${CAMP_DIR}/lib/cmake/camp)
    endif()
    
    if(NOT EXISTS ${_CAMP_SEARCH_PATH})
        message(FATAL_ERROR "Could not find Camp CMake include file (${_CAMP_SEARCH_PATH})")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(camp REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${_CAMP_SEARCH_PATH})
endif()

###############################################################################
# Setup Adios2
###############################################################################
if(NOT ADIOS2_DIR)
    set(ADIOS2_DIR ${ASCENT_ADIOS2_DIR})
endif()

if(ADIOS2_DIR)
    if(NOT EXISTS ${ADIOS2_DIR})
      message(FATAL_ERROR "Could not find ADIOS2 CMake include info (${ADIOS2_DIR}/lib/cmake/adios2)")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(ADIOS2 REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${ADIOS2_DIR})
endif()

###############################################################################
# Setup Fides
###############################################################################
if(NOT FIDES_DIR)
    set(FIDES_DIR ${ASCENT_FIDES_DIR})
endif()

if(FIDES_DIR)
    if(NOT EXISTS ${FIDES_DIR}/lib/cmake/fides)
        message(FATAL_ERROR "Could not find FIDES CMake include info (${FIDES_DIR}/lib/cmake/fides)")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(Fides REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${FIDES_DIR}/lib/cmake/fides)
endif()

###############################################################################
# Setup BabelFlow
###############################################################################
if(NOT BABELFLOW_DIR)
    set(BABELFLOW_DIR ${ASCENT_BABELFLOW_DIR})
endif()

if(BABELFLOW_DIR)
    if(NOT EXISTS ${BABELFLOW_DIR}/lib/cmake/)
        message(FATAL_ERROR "Could not find BabelFLow CMake include info (${BABELFLOW_DIR}/lib/cmake/)")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(BabelFlow REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${BABELFLOW_DIR}/lib/cmake/)
endif()

###############################################################################
# Setup PMT
###############################################################################
if(NOT PMT_DIR)
    set(PMT_DIR ${ASCENT_PMT_DIR})
endif()

if(PMT_DIR)
    if(NOT EXISTS ${PMT_DIR}/lib/cmake)
        message(FATAL_ERROR "Could not find PMT CMake include info (${PMT_DIR}/lib/cmake)")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(PMT REQUIRED
                    NO_DEFAULT_PATH
                    PATHS  ${PMT_DIR}/lib/cmake)
endif()


###############################################################################
# Setup GenTen
###############################################################################
if(NOT GENTEN_DIR)
    set(GENTEN_DIR ${ASCENT_GENTEN_DIR})
endif()

if(GENTEN_DIR)
    if(NOT EXISTS ${GENTEN_DIR}/lib64/cmake/)
        message(FATAL_ERROR "Could not find GenTent CMake include info (${GENTEN_DIR}/lib64/cmake/)")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(Genten REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${GENTEN_DIR}/lib64/cmake/)
endif()

###############################################################################
# MFEM (even serial) may require mpi, if so we need to find mpi
###############################################################################
if(ASCENT_MFEM_MPI_ENABLED AND NOT MPI_FOUND)
    find_package(MPI COMPONENTS CXX)
endif()


###############################################################################
# OCCA + CUDA will require targets from CUDAToolkit
###############################################################################
if(ASCENT_CUDA_ENABLED AND ASCENT_OCCA_ENABLED)
    find_package(CUDAToolkit REQUIRED)
endif()



