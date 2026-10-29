#!/bin/bash -l

export IGC_FunctionCloningThreshold=1
export IGC_ControlInlineTinySize=100
export IGC_OCLInlineThreshold=200
export IGC_PartitionUnit=1
export IGC_ForceOCLSIMDWidth=16
export ZE_AFFINITY_MASK=0.0

# Proxies
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# Use the Aurora default stack (oneAPI 2026.x, libsycl.so.9) instead of
# pinning an eng-compiler: the SYCL build path is validated against the
# default stack, and the resulting build works under a plain default env.
# A pinned compiler rots as ALCF updates the default (the previous
# eng-compiler/2023.12.15.002 pin predates the current SYCL runtime ABI).
# Individual flags can still be overridden via env, e.g.:
#   env build_config=Debug enable_python=OFF ./build_ascent_aurora.sh
module reset
module use /soft/modulefiles
module load cmake

# Python 3.12 + numpy for enable_python=ON, and mpi4py for
# t_python_ascent_mpi_render (its PYTHONPATH is snapshotted from the env
# at configure time, so mpi4py must be loaded before running cmake).
module load python/3.12.12
module load py-cython py-numpy py-pip py-wheel py-setuptools
module load py-mpi4py

env CC=`which mpicc` CXX=`which mpicxx` FTN=`which mpifort` \
  build_config=Release \
  enable_sycl=ON enable_mpi=ON enable_find_mpi=OFF enable_python=ON \
  enable_fortran=ON raja_enable_vectorization=ON enable_tests=ON \
  enable_verbose=ON ./build_ascent_sycl.sh
