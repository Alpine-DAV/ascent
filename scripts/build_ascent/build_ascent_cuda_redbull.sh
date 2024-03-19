#!/bin/bash -l

export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)

env enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF enable_tests=OFF build_ascent=false ./build_ascent_cuda.sh
