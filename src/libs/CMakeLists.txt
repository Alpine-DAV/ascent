# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

################################
# Add ap comp
################################
if(ENABLE_APCOMP)
    add_subdirectory(apcomp)
endif()

################################
# Add dray
################################
if(ENABLE_DRAY)
    add_subdirectory(dray)
endif()


################################
# Add VTK-h
################################
if(ENABLE_VTKH)
    add_subdirectory(vtkh)
endif()

if(ENABLE_VTKH)
  add_subdirectory(rover)
endif()
