#!/bin/bash
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################
set -ev

##########################################################
# test our installed python example
##########################################################

if [ "${ENABLE_PYTHON}" == "ON" ]; then
    if [ "${BUILD_SHARED_LIBS}" == "ON" ]; then
        cd ${TRAVIS_BUILD_DIR}/travis-debug-install/
        env LD_LIBRARY_PATH=${RUN_LIB_PATH} ./bin/run_python_with_ascent.sh < examples/ascent/python/ascent_python_render_example.py
    fi
fi


