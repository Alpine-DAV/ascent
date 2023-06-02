module load cmake

# Swap NVHPC with GNU compilers
module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

export CC=$(which gcc)
export CXX=$(which g++)
export FTN=$(which gfortran)

env enable_mpi=ON ./build_ascent_cuda.sh