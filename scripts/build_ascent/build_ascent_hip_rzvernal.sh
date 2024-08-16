module load cmake/3.24.2
module load craype-accel-amd-gfx90a
module load rocmcc/6.1.2-magic

export MPICH_GPU_SUPPORT_ENABLED=1
export ROCM_ARCH=gfx90a

export CC=/opt/rocm-6.1.2/bin/amdclang
export CXX=/opt/rocm-6.1.2/bin/amdclang++
#export CFLAGS="-I${ROCM_PATH}/include"
#export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed"
#export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"

export enable_python="${enable_python:=ON}"
export build_caliper="${build_caliper:=true}"
export build_pyvenv="${bbuild_pyvenv:=true}"

./build_ascent_hip.sh


