#!/bin/bash
module load cmake/3.23.1
module load gcc/7.3.1
module load cuda/11.2.0

export CUDA_ARCH=70
export CUDA_ARCH_VTKM=volta

export CC=$(which gcc)
export CXX=$(which g++)
export FTN=$(which gfortran)

env  enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF enable_tests=OFF ./build_ascent_cuda.sh
