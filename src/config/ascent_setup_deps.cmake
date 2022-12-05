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
# HIP related tpls will require targets from hip
###############################################################################
if(ASCENT_HIP_ENABLED)
    ####################################
    # IMPORANT NOTE AND FUN CMAKE FACT: 
    ####################################
    # The HIP CMake Pacakge *requires* ROCM_PATH to be set.
    #
    # If not set, it won't find other reqd cmake imports (like AMDDeviceLibs)
    #
    # You *cannot* just hand the path as an arg like ${ASCENT_ROCM_PATH}
    # to find_package, ROCM_PATH must be set.
    #
    if(NOT ROCM_PATH)
        set(ROCM_PATH ${ASCENT_ROCM_PATH})
    endif()
    find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH})
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
# Setup Caliper
###############################################################################
if(NOT CALIPER_DIR)
    set(CALIPER_DIR ${ASCENT_CALIPER_DIR})
endif()

if(CALIPER_DIR)
    if(NOT Ascent_FIND_QUIETLY)
        message(STATUS "Ascent was built with Caliper Support")
    endif()

    if(NOT ADIAK_DIR)
        set(ADIAK_DIR ${ASCENT_ADIAK_DIR})
    endif()

    if(ADIAK_DIR)
        if(NOT Ascent_FIND_QUIETLY)
            message(STATUS "Looking for Adiak at: ${ADIAK_DIR}/lib/cmake/adiak")
        endif()
        # find adiak first
        find_package(adiak REQUIRED
                     NO_DEFAULT_PATH
                     PATHS ${ADIAK_DIR}/lib/cmake/adiak)
    endif()
    if(NOT Ascent_FIND_QUIETLY)
        message(STATUS "Looking for Caliper at: ${CALIPER_DIR}/share/cmake/caliper")
    endif()
    # find caliper
    find_package(caliper REQUIRED
                 NO_DEFAULT_PATH
                 PATHS ${CALIPER_DIR}/share/cmake/caliper)
endif()

###############################################################################
# Setup Kokkos
###############################################################################
if(NOT KOKKOS_DIR)
    set(KOKKOS_DIR ${ASCENT_KOKKOS_DIR})
endif()

if(EXISTS ${KOKKOS_DIR}/lib64/cmake/Kokkos/)
    set(KOKKOS_CMAKE_CONFIG_DIR ${KOKKOS_DIR}/lib64/cmake/Kokkos/)
endif()

if(EXISTS ${KOKKOS_DIR}/lib/cmake/Kokkos/)
    set(KOKKOS_CMAKE_CONFIG_DIR ${KOKKOS_DIR}/lib/cmake/Kokkos/)
endif()


if(KOKKOS_DIR)
    if(NOT EXISTS ${KOKKOS_CMAKE_CONFIG_DIR}/KokkosConfig.cmake)
        MESSAGE(FATAL_ERROR "Could not find Kokkos CMake include file (${KOKKOS_CMAKE_CONFIG_DIR}/KokkosConfig.cmake)")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(Kokkos REQUIRED
                    NO_DEFAULT_PATH
		    PATHS ${KOKKOS_CMAKE_CONFIG_DIR})
endif()

###############################################################################
# Setup VTK-m
###############################################################################
if(NOT VTKM_DIR)
    set(VTKM_DIR ${ASCENT_VTKM_DIR})
endif()

if(VTKM_DIR)
    # use VTKM_DIR to setup the options that cmake's find VTKm needs
    if(NOT EXISTS ${VTKM_DIR})
        message(FATAL_ERROR "Failed to find VTKm at VTKM_DIR=${VTKM_DIR}")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(VTKm REQUIRED
      NO_DEFAULT_PATH
      PATHS ${VTKM_DIR})
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
# Setup RAJA
###############################################################################
if(NOT RAJA_DIR)
    set(RAJA_DIR ${ASCENT_RAJA_DIR})
endif()

if(RAJA_DIR)
    set(_RAJA_SEARCH_PATH)
    if(EXISTS ${RAJA_DIR}/share/raja/cmake)
      # old install layout
      set(_RAJA_SEARCH_PATH ${RAJA_DIR}/share/raja/cmake)
    else()
      # new install layout
      set(_RAJA_SEARCH_PATH ${RAJA_DIR}/lib/cmake/raja)
    endif()
    
    if(NOT EXISTS ${_RAJA_SEARCH_PATH})
        message(FATAL_ERROR "Could not find RAJA CMake include file (${_RAJA_SEARCH_PATH})")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(RAJA REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${_RAJA_SEARCH_PATH})
endif()

###############################################################################
# Setup Adios2
###############################################################################
if(NOT ADIOS2_DIR)
    set(ADIOS2_DIR ${ASCENT_ADIOS2_DIR})
endif()

if(ADIOS2_DIR)
    if(NOT EXISTS ${ADIOS2_DIR})
      message(FATAL_ERROR "Could not find ADIOS2 CMake include info (${ADIOS2_DIR})")
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
    if(NOT EXISTS ${FIDES_DIR})
        message(FATAL_ERROR "Could not find FIDES CMake include info (${FIDES_DIR})")
    endif()

    ###############################################################################
    # Import CMake targets
    ###############################################################################
    find_dependency(Fides REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${FIDES_DIR})
endif()

###############################################################################
# Setup BabelFlow
###############################################################################
if(ASCENT_BABELFLOW_ENABLED)
    ##########################################################################
    # BabelFlow
    ##########################################################################
    if(NOT BABELFLOW_DIR)
        set(BABELFLOW_DIR ${ASCENT_BABELFLOW_DIR})
    endif()

    if(BABELFLOW_DIR)
        if(NOT EXISTS ${BABELFLOW_DIR}/lib/cmake/)
            message(FATAL_ERROR "Could not find BabelFLow CMake include info (${BABELFLOW_DIR}/lib/cmake/)")
        endif()

        ######################################################################
        # Import CMake targets
        ######################################################################
        find_dependency(BabelFlow REQUIRED
                        NO_DEFAULT_PATH
                        PATHS ${BABELFLOW_DIR}/lib/cmake/)
    endif()

    ##########################################################################
    # Setup PMT
    ##########################################################################
    if(NOT PMT_DIR)
        set(PMT_DIR ${ASCENT_PMT_DIR})
    endif()

    if(PMT_DIR)
        if(NOT EXISTS ${PMT_DIR}/lib/cmake)
            message(FATAL_ERROR "Could not find PMT CMake include info (${PMT_DIR}/lib/cmake)")
        endif()

        ######################################################################
        # Import CMake targets
        ######################################################################
        find_dependency(PMT REQUIRED
                        NO_DEFAULT_PATH
                        PATHS  ${PMT_DIR}/lib/cmake)
    endif()


    ##########################################################################
    # Setup StreamStat
    ##########################################################################
    if(NOT STREAMSTAT_DIR)
        set(STREAMSTAT_DIR ${ASCENT_STREAMSTAT_DIR})
    endif()

    if(STREAMSTAT_DIR)
        if(NOT EXISTS ${STREAMSTAT_DIR}/lib/cmake)
            message(FATAL_ERROR "Could not find StreamStat CMake include info (${STREAMSTAT_DIR}/lib/cmake)")
        endif()

        ######################################################################
        # Import CMake targets
        ######################################################################
        find_dependency(StreamStat REQUIRED
                        NO_DEFAULT_PATH
                        PATHS  ${STREAMSTAT_DIR}/lib/cmake)
    endif()

    ##########################################################################
    # Setup TopoFileParser
    ##########################################################################
    if(NOT TOPOFILEPARSER_DIR)
        set(TOPOFILEPARSER_DIR ${ASCENT_TOPOFILEPARSER_DIR})
    endif()

    if(TOPOFILEPARSER_DIR)
        if(NOT EXISTS ${TOPOFILEPARSER_DIR}/lib/cmake)
            message(FATAL_ERROR "Could not find TopoFileParser CMake include info (${TOPOFILEPARSER_DIR}/lib/cmake)")
        endif()

        ######################################################################
        # Import CMake targets
        ######################################################################
        find_dependency(TopoFileParser REQUIRED
                        NO_DEFAULT_PATH
                        PATHS  ${TOPOFILEPARSER_DIR}/lib/cmake)
    endif()
endif() # end if babelflow

###############################################################################
# Setup GenTen
###############################################################################
if(NOT GENTEN_DIR)
    set(GENTEN_DIR ${ASCENT_GENTEN_DIR})
endif()

if(GENTEN_DIR)
    if(NOT EXISTS ${GENTEN_DIR}/lib64/cmake/)
        message(FATAL_ERROR "Could not find GenTen CMake include info (${GENTEN_DIR}/lib64/cmake/)")
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


