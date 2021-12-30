###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# BabelFlow iso surfaces example 
# Usage: mpirun -n N ./ascent_bflow_iso_example <dim> <fanin> <blk x> <blk y> <blk z> <radices>
#
##############################################################################


set(ascent_bflow_iso_example_sources
    ascent_bflow_iso_example.cpp)

if(MPI_FOUND)
  set(ascent_bflow_iso_example_deps ascent_mpi BabelFlow::babelflow ${MPI_CXX_LIBRARIES})

    blt_add_executable(
        NAME        ascent_bflow_iso_example
        SOURCES     ${ascent_bflow_iso_example_sources}
        DEPENDS_ON  ${ascent_bflow_iso_example_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

  install(TARGETS ascent_bflow_iso_example
      EXPORT  ascent
      LIBRARY DESTINATION examples/ascent/babelflow/iso
      ARCHIVE DESTINATION examples/ascent/babelflow/iso
      RUNTIME DESTINATION examples/ascent/babelflow/iso)
endif()
