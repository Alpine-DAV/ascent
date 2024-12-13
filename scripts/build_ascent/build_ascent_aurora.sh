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

module reset
module use /soft/modulefiles
module load spack-pe-gcc cmake
module load oneapi/eng-compiler/2023.12.15.002

env CC=`which icx` CXX=`which icpx` FTN=`which ifx` enable_sycl=ON enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF enable_tests=ON enable_verbose=ON ./build_ascent_sycl.sh
