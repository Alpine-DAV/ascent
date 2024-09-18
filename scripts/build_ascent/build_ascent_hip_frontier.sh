module load cmake #3.23.2
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm/5.7.1
module load cray-mpich/8.1.28
module load cce/17.0.0
module load cray-python/3.11.5


export MPICH_GPU_SUPPORT_ENABLED=1
export ROCM_ARCH=gfx90a
export CC=$(which cc)
export CXX=$(which CC)
export FTN=$(which ftn)
export CFLAGS="${CRAY_ROCM_INCLUDE_OPTS} -I/opt/cray/pe/mpich/8.1.28/ofi/crayclang/17.0/include/"
export CXXFLAGS="${CRAY_ROCM_INCLUDE_OPTS} -I/opt/cray/pe/mpich/8.1.28/ofi/crayclang/17.0/include/ -Wno-pass-failed"
export LDFLAGS="${CRAY_ROCM_POST_LINK_OPTS}"
#export HIPFLAGS="-I/opt/cray/pe/mpich/default/ofi/rocm-compiler/5.0/include/"
env enable_mpi=ON enable_find_mpi=OFF build_pyvenv=true ENABLE_PYTHON=ON ./build_ascent_hip.sh


