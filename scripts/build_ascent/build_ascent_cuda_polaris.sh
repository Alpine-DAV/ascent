#!/bin/bash -l
module load cmake

# Swap NVHPC with GNU compilers
module swap PrgEnv-nvhpc PrgEnv-gnu
module load gcc/11.2.0
module load cudatoolkit-standalone

export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)

env enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF enable_tests=OFF ./build_ascent_cuda.sh
