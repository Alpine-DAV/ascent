###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

if(NOT VTKH_DIR)
  MESSAGE(FATAL_ERROR "VTKh support needs explicit VTKH_DIR")
endif()

MESSAGE(STATUS "Looking for VTKh using VTKH_DIR = ${VTKH_DIR}")

set(VTKh_DIR ${VTKH_DIR}/lib)

find_package(VTKh REQUIRED)
message(STATUS "Found VTKh include dirs: ${VTKh_INCLUDE_DIRS}")

set(VTKH_FOUND TRUE)


blt_register_library(NAME vtkh
                     INCLUDES ${VTKh_INCLUDE_DIRS}
                     LIBRARIES vtkh)

if (MPI_FOUND)
    blt_register_library(NAME vtkh_mpi
                         INCLUDES ${VTKh_INCLUDE_DIRS}
                         LIBRARIES vtkh_mpi)

endif()
