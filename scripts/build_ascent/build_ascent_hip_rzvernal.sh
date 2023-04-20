module load cmake/3.24.2
module load craype-accel-amd-gfx90a
module load rocm/5.2.3
module load cray-mpich

export MPICH_GPU_SUPPORT_ENABLED=1
export ROCM_ARCH=gfx90a
#export CC=$(which cc)
#export CXX=$(which CC)
#export FTN=$(which ftn)

export CC=/opt/rocm-5.2.3/llvm/bin/amdclang
export CXX=/opt/rocm-5.2.3/llvm/bin/amdclang++
export CFLAGS="-I${ROCM_PATH}/include"
export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed"
export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
./build_ascent_hip.sh


