###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

################################
# Guards for Fortran support.
################################
if(ENABLE_FORTRAN)
    if(CMAKE_Fortran_COMPILER)
        MESSAGE(STATUS  "Fortran Compiler: ${CMAKE_Fortran_COMPILER}")
        set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/fortran)
    elseif(CMAKE_GENERATOR STREQUAL Xcode)
        MESSAGE(STATUS "Disabling Fortran support: ENABLE_FORTRAN is true, "
                       "but the Xcode CMake Generator does not support Fortran.")
        set(ENABLE_FORTRAN OFF)
    else()
        MESSAGE(FATAL_ERROR "ENABLE_FORTRAN is true, but a Fortran compiler wasn't found.")
    endif()
    set(FORTRAN_FOUND 1)
endif()

