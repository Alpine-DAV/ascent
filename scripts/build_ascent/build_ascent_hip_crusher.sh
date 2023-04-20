module load cmake/3.23.2
module load craype-accel-amd-gfx90a
module load rocm/5.2.0
module load cray-mpich/8.1.16

export MPICH_GPU_SUPPORT_ENABLED=1
export ROCM_ARCH=gfx90a
export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)
export CFLAGS="-I${ROCM_PATH}/include"
export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed"
export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
export HIPFLAGS="-I/opt/cray/pe/mpich/default/ofi/rocm-compiler/5.0/include/"
env enable_mpi=ON enable_find_mpi=OFF ./build_ascent_hip.sh


