###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# Setup Fides
#
###############################################################################

if(NOT FIDES_DIR)
    MESSAGE(FATAL_ERROR "Fides support needs explicit FIDES_DIR")
endif()

if(NOT VTKM_DIR)
    MESSAGE(FATAL_ERROR "Fides support needs VTK-m (VTKM_DIR not set)")
endif()

if(NOT ADIOS2_DIR)
    MESSAGE(FATAL_ERROR "Fides support needs ADIOS2 (ADIOS2_DIR not set)")
endif()

MESSAGE(STATUS "Looking for FIDES using FIDES_DIR = ${FIDES_DIR}")

set(FIDES_DIR_ORIG ${FIDES_DIR})

#The Fides cmake is not setting these for some reason.
#So, we set them explicitly for now.
#set(Fides_DIR ${FIDES_DIR})

set(FIDES_INCLUDE_DIR ${FIDES_DIR}/include/)
set(FIDES_LIB_DIR ${FIDES_DIR}/lib)
set(FIDES_LIBRARIES fides)

find_package(Fides REQUIRED
             NO_DEFAULT_PATH
             PATHS ${FIDES_DIR}/lib/cmake/fides)


message(STATUS "Found Fides at ${FIDES_DIR}")
set(FIDES_FOUND TRUE)

# Fides cmake detection logic can change ADIOS2_FOUND to 1
# so we reset it here to be consistent with other libs
set(ADIOS2_FOUND TRUE)


blt_register_library(NAME fides
                     INCLUDES ${FIDES_INCLUDE_DIR}
                     LIBRARIES ${FIDES_LIB_DIRS} ${FIDES_LIBRARIES} )
