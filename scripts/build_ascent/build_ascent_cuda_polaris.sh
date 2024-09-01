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

# export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
# export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
# export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
# source /home/qiwu/projects/diva_superbuild/setup-env.sh

export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)

# env enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF enable_tests=OFF ./build_ascent_cuda.sh
# env enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF enable_tests=OFF build_ascent=false ./build_ascent_cuda.sh
# env enable_python=ON enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF enable_tests=OFF ./build_ascent_cuda.sh
env enable_python=ON enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF enable_tests=OFF build_ascent=false ./build_ascent_cuda.sh
