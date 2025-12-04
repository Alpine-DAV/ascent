#!/bin/bash -l

if [[ ! "${HOSTNAME}" =~ aurora-uan ]]; then
  export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
  export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
  export http_proxy="http://proxy.alcf.anl.gov:3128"
  export https_proxy="http://proxy.alcf.anl.gov:3128"
  export ftp_proxy="http://proxy.alcf.anl.gov:3128"
  export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
fi

module reset
module use /soft/modulefiles
module load cmake
module load python/3.10.14
module load py-cython py-numpy py-pip py-wheel py-setuptools

#env CC=`which icx` CXX=`which icpx` FTN=`which ifx` enable_sycl=ON enable_mpi=ON enable_find_mpi=OFF enable_fortran=ON enable_python=ON raja_enable_vectorization=OFF enable_tests=OFF enable_verbose=OFF install_dir=/soft/visualization/ascent/release/v0.9.4 ./build_ascent_sycl.sh
env CC=`which mpicc` CXX=`which mpicxx` FTN=`which mpifort` enable_64bit_ids=ON enable_sycl=ON enable_mpi=ON enable_find_mpi=OFF enable_fortran=ON enable_python=ON raja_enable_vectorization=ON enable_tests=OFF enable_verbose=OFF install_dir=/soft/visualization/ascent/release/v0.9.5 ./build_ascent_sycl.sh
#env CC=`which icx` CXX=`which icpx` FTN=`which ifx` env mpicc_exe=`which mpicc` mpicxx_exe=`which mpicxx` enable_sycl=ON enable_mpi=ON enable_find_mpi=OFF enable_mpicc=ON enable_fortran=ON enable_python=ON raja_enable_vectorization=ON enable_tests=OFF enable_verbose=OFF ./build_ascent_sycl.sh
