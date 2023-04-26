# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

################################
# Ascent 3rd Party Dependencies
################################

###############################################################################
# gtest, fruit, mpi, cuda, openmp, sphinx and doxygen are handled by blt
###############################################################################

################################
# Setup Python if requested
################################
if(ENABLE_PYTHON)
    include(cmake/thirdparty/SetupPython.cmake)
    message(STATUS "Using Python Include: ${PYTHON_INCLUDE_DIRS}")
    include_directories(${PYTHON_INCLUDE_DIRS})
    # if we don't find python, throw a fatal error
    if(NOT PYTHON_FOUND)
        message(FATAL_ERROR "ENABLE_PYTHON is true, but Python wasn't found.")
    endif()
endif()

# clear DLL runtime path
set(ASCENT_TPL_DLL_PATHS)

################################
# Conduit
################################
include(cmake/thirdparty/SetupConduit.cmake)

################################################################
################################################################
#
# 3rd Party Libs that underpin Ascent's Pipelines
#
################################################################
################################################################

################################
# Caliper
################################
if(CALIPER_DIR)
    include(cmake/thirdparty/SetupCaliper.cmake)
endif()

################################
# Camp
################################
if(CAMP_DIR) # optional for now
    include(cmake/thirdparty/SetupCamp.cmake)
endif()

################################
# Umpire
################################
if(UMPIRE_DIR) # optional for now
    include(cmake/thirdparty/SetupUmpire.cmake)
endif()

################################
# RAJA
################################
if(RAJA_DIR)  # optional for now
    include(cmake/thirdparty/SetupRAJA.cmake)
endif()

################################
# Setup Kokkos
################################
if (KOKKOS_DIR)
  include(cmake/thirdparty/SetupKokkos.cmake)
endif()

################################
# Setup OCCA
################################
if (OCCA_DIR)
  include(cmake/thirdparty/SetupOcca.cmake)
endif()

################################
# VTKm and supporting libs
################################
if(VTKM_DIR)
    include(cmake/thirdparty/SetupVTKm.cmake)
endif()

if(ENABLE_VTKH AND NOT VTKM_FOUND)
     MESSAGE(FATAL_ERROR "VTK-h support requires VTK-m (ENABLE_VTKH=ON and NOT VTKM_FOUND)")
endif()

#
# Note: HDF5 is fully handled by importing conduit
#

################################
# Setup MFEM if enabled
################################
if (MFEM_DIR)
  include(cmake/thirdparty/SetupMFEM.cmake)
endif()

################################
# Setup Umpire
################################
if (UMPIRE_DIR)
  include(cmake/thirdparty/SetupUmpire.cmake)
endif()

################################
# Setup ADIOS2
################################
if (ADIOS2_DIR)
  include(cmake/thirdparty/SetupADIOS2.cmake)
endif()

################################
# Setup Fides
################################
if (FIDES_DIR)
  include(cmake/thirdparty/SetupFides.cmake)
endif()

################################
# Setup Babelflow
################################
if (BABELFLOW_DIR OR BabelFlow_DIR)
  include(cmake/thirdparty/SetupBabelFlow.cmake)
endif()


################################
# Setup GenTen
################################
if (GENTEN_DIR)
  include(cmake/thirdparty/SetupGenTen.cmake)
endif()
