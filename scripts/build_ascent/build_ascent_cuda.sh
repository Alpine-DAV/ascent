#!/bin/bash

##############################################################################
# Demonstrates how to manually build Ascent and its dependencies, including:
#
#  hdf5, conduit, vtk-m, mfem, raja, and umpire
#
# usage example:
#   env enable_mpi=ON enable_openmp=ON ./build_ascent.sh
#
#
# Assumes: 
#  - cmake is in your path
#  - selected compilers (including nvcc) are in your path or set via env vars
#  - [when enabled] MPI and Python (+numpy and mpi4py), are in your path
#
##############################################################################
set -eu -o pipefail

# 2023-04-21 CUDA support is handled by our unified script
env enable_cuda=ON ./build_ascent.sh
