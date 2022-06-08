# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

add_subdirectory(color_tables)
add_subdirectory(furnace)
add_subdirectory(chopper)
add_subdirectory(subset)


if(MFEM_FOUND)
    add_subdirectory(mfem_solve)
endif()
