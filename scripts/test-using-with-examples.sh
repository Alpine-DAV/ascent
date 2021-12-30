#!/bin/bash
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

# using with cmake example
cd ${ASCENT_DIR}/examples/ascent/using-with-cmake
mkdir build
cd build
export EXAMPLE_CFG="-DASCENT_DIR=${ASCENT_DIR} -DCONDUIT_DIR=${CONDUIT_DIR} -DVTKM_DIR=${VTKM_DIR} -DVTKH_DIR=${VTKH_DIR}"
cmake  ${EXAMPLE_CFG} ../
make VERBOSE=1
./ascent_render_example

# using with make example
cd ${TRAVIS_BUILD_DIR}/src/examples/using-with-make
make
env LD_LIBRARY_PATH=${ASCENT_DIR}/lib/:${CONDUIT_DIR}/lib/ ./ascent_render_example

