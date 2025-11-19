#!/bin/bash -l

module reset
module use /soft/modulefiles
module load PrgEnv-gnu
module load gcc-native/12.3.lua
module load cudatoolkit-standalone/12.9.1
module load cuda/12.6
module load craype-accel-nvidia80
module load spack-pe-base cmake
module load cray-python

module list

export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)

env enable_python=ON enable_mpi=ON enable_fortran=OFF raja_enable_vectorization=OFF enable_tests=OFF ./build_ascent_cuda.sh

