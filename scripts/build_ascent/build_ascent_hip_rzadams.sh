module load cmake/3.24.2
module load craype-accel-amd-gfx940
module load rocmcc/6.1.2-magic

export MPICH_GPU_SUPPORT_ENABLED=1
export ROCM_ARCH=gfx942

export CC=/usr/tce/packages/rocmcc/rocmcc-6.1.2-magic/bin/amdclang
export CXX=/usr/tce/packages/rocmcc/rocmcc-6.1.2-magic/bin/amdclang++
export ROCM_PATH=/usr/tce/packages/rocmcc/rocmcc-6.1.2-magic/

export enable_mpi="${enable_mpi:=ON}"
export enable_python="${enable_python:=ON}"
export build_caliper="${build_caliper:=true}"
export build_pyvenv="${bbuild_pyvenv:=true}"

./build_ascent_hip.sh


