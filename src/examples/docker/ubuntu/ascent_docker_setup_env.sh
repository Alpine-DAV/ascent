#!/bin/bash
####################################
# env setup for Ascent docker build
####################################
# to use run:
# >source ascent_docker_setup.sh
####################################

# setup paths to spack built python and mpich
export PATH=`ls -d /ascent/uberenv_libs/spack/opt/spack/*/*/mpich*/bin`:$PATH
export PATH=`ls -d /ascent/uberenv_libs/spack/opt/spack/*/*/python*/bin`:$PATH
# add the ascent python module to the python path
export PYTHONPATH=/ascent/install-debug/python-modules/