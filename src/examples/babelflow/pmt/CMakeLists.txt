###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# BabelFlow PMT example that reads an output file
# Usage: mpirun -n N ./ascent_babelflow_example -f <input_data> 
# 		-d <Xdim> <Ydim> <Zdim> -p <dx> <dy> <dz> -m <fanin> -t <threshold>
#
##############################################################################


set(ascent_babelflow_example_sources
    ascent_babelflow_example.cpp)

set(ascent_babelflow_example_header)

if(MPI_FOUND)
	set(ascent_babelflow_example_deps ascent_mpi BabelFlow::babelflow ${MPI_CXX_LIBRARIES} PMT::pmt)

    blt_add_executable(
        NAME        ascent_babelflow_example
        SOURCES     ${ascent_babelflow_example_sources}
        DEPENDS_ON  ${ascent_babelflow_example_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

  install(TARGETS ascent_babelflow_example
      EXPORT  ascent
      LIBRARY DESTINATION examples/ascent/babelflow/pmt
      ARCHIVE DESTINATION examples/ascent/babelflow/pmt
      RUNTIME DESTINATION examples/ascent/babelflow/pmt)
endif()
