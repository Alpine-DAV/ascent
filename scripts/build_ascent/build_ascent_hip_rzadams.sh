module load cmake/3.24.2
module load craype-accel-amd-gfx940
module load rocmcc/6.4.3leakfix-magic

export MPICH_GPU_SUPPORT_ENABLED=1
export ROCM_ARCH=gfx942

export CC=/usr/tce/packages/rocmcc/rocmcc-6.4.3leakfix-magic/bin/amdclang
export CXX=/usr/tce/packages/rocmcc/rocmcc-6.4.3leakfix-magic/bin/amdclang++
export cxx_standard=20
export ROCM_PATH=/usr/tce/packages/rocmcc/rocmcc-6.4.3leakfix-magic/

export enable_mpi="${enable_mpi:=ON}"
export enable_python="${enable_python:=OFF}"
export build_caliper="${build_caliper:=true}"
export build_pyvenv="${build_pyvenv:=false}"
export build_zfp="${build_zfp:=false}"
export build_shared_libs="${build_shared_libs:=false}"

./build_ascent_hip.sh
