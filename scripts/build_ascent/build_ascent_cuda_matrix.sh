module load gcc/12.1.1
module load cuda/12.6.0
module load cmake/3.30.5
module load mvapich2/2.3.7

module list

export CUDA_ARCH=90
export CUDA_ARCH_VISKORES=hopper

export CC="${CC:=$(which gcc)}"
export CXX="${CXX:=$(which g++)}"
export FTN="${FTN:=$(which gfortran)}"

env enable_mpi=ON enable_tests=ON enable_find_mpi=ON enable_fortran=OFF ./build_ascent_cuda.sh
