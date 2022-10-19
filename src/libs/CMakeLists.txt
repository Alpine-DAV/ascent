# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.


################################
# Add flow
################################
add_subdirectory(flow)


# display info about optional features
if(ENABLE_APCOMP)
    message(STATUS "Building APComp    (ENABLE_APCOMP=ON)")
endif()

if(ENABLE_DRAY)
    message(STATUS "Building Devil Ray (ENABLE_DRAY=ON)")
endif()

if((ENABLE_VTKH))
    message(STATUS "Building VTK-h     (ENABLE_VTKH=ON)")
endif()


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
    add_subdirectory(rover)
endif()
