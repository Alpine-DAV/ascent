###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# file: src/examples/CMakeLists.txt
#
###############################################################################
add_subdirectory(proxies/lulesh2.0.3)
add_subdirectory(proxies/kripke)
add_subdirectory(proxies/cloverleaf3d-ref)
add_subdirectory(synthetic/noise)
add_subdirectory(paraview-vis)

if(MFEM_FOUND AND MFEM_MPI_ENABLED)
  add_subdirectory(proxies/laghos)
endif()

if(BABELFLOW_FOUND)
  add_subdirectory(babelflow/pmt)
  add_subdirectory(babelflow/comp)
  add_subdirectory(babelflow/iso)
endif()

# install using and python examples
install(DIRECTORY using-with-cmake using-with-make python tutorial 
        DESTINATION examples/ascent)

if(MPI_FOUND)
    install(DIRECTORY using-with-cmake-mpi using-with-make-mpi
            DESTINATION examples/ascent)
endif()

