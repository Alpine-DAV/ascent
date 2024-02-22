# module load cmake
# # Swap NVHPC with GNU compilers
# module swap PrgEnv-nvhpc PrgEnv-gnu
# module load cudatoolkit-standalone

# export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
# export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
# export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"

# source /home/qiwu/projects/diva_superbuild/setup-env.sh

export CC=$(which gcc)
export CXX=$(which g++)
export FTN=$(which gfortran)

env enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF build_ascent=false CUDA_ARCH=75 ./build_ascent_cuda.sh
