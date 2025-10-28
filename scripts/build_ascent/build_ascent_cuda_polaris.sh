#!/bin/bash -l
module use /soft/modulefiles
module load PrgEnv-gnu
module load nvhpc-mixed
module load craype-accel-nvidia80
module unload nvhpc-mixed
module load spack-pe-base cmake
module load cudatoolkit-standalone
module load craype-x86-milan
module load cray-python

export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)

env enable_python=ON enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF enable_tests=OFF ./build_ascent_cuda.sh
