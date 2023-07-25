# module load cmake
# # Swap NVHPC with GNU compilers
# module swap PrgEnv-nvhpc PrgEnv-gnu
# module load cudatoolkit-standalone

export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"

source /home/qiwu/projects/diva_superbuild/setup-env.sh

export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)

env enable_mpi=ON enable_fortran=ON raja_enable_vectorization=OFF build_ascent=false ./build_ascent_cuda.sh
