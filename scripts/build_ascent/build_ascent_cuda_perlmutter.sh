module load cmake/3.24.3

export CUDA_ARCH=80
export CUDA_ARCH_VTKM=ampere

export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)
env enable_mpi=ON enable_find_mpi=OFF ./build_ascent_cuda.sh

