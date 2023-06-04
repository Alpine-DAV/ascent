module load cmake

# Swap NVHPC with GNU compilers
module swap PrgEnv-nvhpc PrgEnv-gnu
module load cudatoolkit-standalone/11.8.0

export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)

env enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF ./build_ascent_cuda.sh
