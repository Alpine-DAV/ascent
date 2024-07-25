#!/bin/bash

##############################################################################
# Demonstrates how to manually build Ascent and its dependencies, including:
#
#  hdf5, conduit, vtk-m, mfem, raja, and umpire
#
# usage example:
#   env enable_mpi=ON enable_openmp=ON ./build_ascent.sh
#
#
# Assumes: 
#  - cmake is in your path
#  - selected compilers are in your path or set via env vars
#  - [when enabled] MPI and Python (+numpy and mpi4py), are in your path
#
##############################################################################
set -eu -o pipefail

##############################################################################
# Build Options
##############################################################################

# shared options
enable_cuda="${enable_cuda:=OFF}"
enable_hip="${enable_hip:=OFF}"
enable_fortran="${enable_fortran:=OFF}"
enable_python="${enable_python:=OFF}"
enable_openmp="${enable_openmp:=OFF}"
enable_mpi="${enable_mpi:=OFF}"
enable_find_mpi="${enable_find_mpi:=ON}"
enable_tests="${enable_tests:=OFF}"
enable_verbose="${enable_verbose:=ON}"
build_jobs="${build_jobs:=6}"
build_config="${build_config:=Release}"
build_shared_libs="${build_shared_libs:=ON}"

# tpl controls
build_zlib="${build_zlib:=true}"
build_hdf5="${build_hdf5:=true}"
build_pyvenv="${build_pyvenv:=false}"
build_silo="${build_silo:=true}"
build_conduit="${build_conduit:=true}"
build_vtkm="${build_vtkm:=true}"
build_camp="${build_camp:=true}"
build_raja="${build_raja:=true}"
build_umpire="${build_umpire:=true}"
build_mfem="${build_mfem:=true}"
build_catalyst="${build_catalyst:=false}"

# ascent options
build_ascent="${build_ascent:=true}"

# see if we are building on windows
build_windows="${build_windows:=OFF}"

# see if we are building on macOS
build_macos="${build_macos:=OFF}"

if [[ "$enable_cuda" == "ON" ]]; then
    echo "*** configuring with CUDA support"

    CC="${CC:=gcc}"
    CXX="${CXX:=g++}"
    FTN="${FTN:=gfortran}"

    CUDA_ARCH="${CUDA_ARCH:=80}"
    CUDA_ARCH_VTKM="${CUDA_ARCH_VTKM:=ampere}"
fi

if [[ "$enable_hip" == "ON" ]]; then
    echo "*** configuring with HIP support"

    CC="${CC:=/opt/rocm/llvm/bin/amdclang}"
    CXX="${CXX:=/opt/rocm/llvm/bin/amdclang++}"
    # FTN?

    ROCM_ARCH="${ROCM_ARCH:=gfx90a}"
    ROCM_PATH="${ROCM_PATH:=/opt/rocm/}"

    # NOTE: this script only builds kokkos when enable_hip=ON
    build_kokkos="${build_kokkos:=true}"
else
    build_kokkos="${build_kokkos:=false}"
fi

case "$OSTYPE" in
  win*)     build_windows="ON";;
  msys*)    build_windows="ON";;
  darwin*)  build_macos="ON";;
  *)        ;;
esac

if [[ "$build_windows" == "ON" ]]; then
  echo "*** configuring for windows"
fi

if [[ "$build_macos" == "ON" ]]; then
  echo "*** configuring for macos"
fi

################
# path helpers
################
function ospath()
{
  if [[ "$build_windows" == "ON" ]]; then
    echo `cygpath -m $1`
  else
    echo $1
  fi 
}

function abs_path()
{
  if [[ "$build_macos" == "ON" ]]; then
    echo "$(cd $(dirname "$1");pwd)/$(basename "$1")"
  else
    echo `realpath $1`
  fi
}

root_dir=$(pwd)
root_dir="${prefix:=${root_dir}}"
root_dir=$(ospath ${root_dir})
root_dir=$(abs_path ${root_dir})
script_dir=$(abs_path "$(dirname "${BASH_SOURCE[0]}")")
build_dir=$(ospath ${root_dir}/build)
source_dir=$(ospath ${root_dir}/source)

# root_dir is where we will build and install
# override with `prefix` env var
if [ ! -d ${root_dir} ]; then
  mkdir -p ${root_dir}
fi

cd ${root_dir}

# install_dir is where we will install
# override with `prefix` env var
install_dir="${install_dir:=$root_dir/install}"

echo "*** prefix:       ${root_dir}" 
echo "*** build root:   ${build_dir}"
echo "*** sources root: ${source_dir}"
echo "*** install root: ${install_dir}"
echo "*** script dir:   ${script_dir}"

################
# tar options
################
tar_extra_args=""
if [[ "$build_windows" == "ON" ]]; then
  tar_extra_args="--force-local"
fi 

# make sure sources dir exists
if [ ! -d ${source_dir} ]; then
  mkdir -p ${source_dir}
fi
################
# CMake Compiler Settings
################
cmake_compiler_settings=""

# capture compilers if they are provided via env vars
if [ ! -z ${CC+x} ]; then
  cmake_compiler_settings="-DCMAKE_C_COMPILER:PATH=${CC}"
fi

if [ ! -z ${CXX+x} ]; then
  cmake_compiler_settings="${cmake_compiler_settings} -DCMAKE_CXX_COMPILER:PATH=${CXX}"
fi

if [ ! -z ${FTN+x} ]; then
  cmake_compiler_settings="${cmake_compiler_settings} -DCMAKE_Fortran_COMPILER:PATH=${FTN}"
fi

################
# print all build_ZZZ and enable_ZZZ options
################
echo "*** cmake_compiler_settings: ${cmake_compiler_settings}"
echo "*** build_ascent `enable` settings:"
set | grep enable_
echo "*** build_ascent `build` settings:"
set | grep build_

################
# Zlib
################
zlib_version=1.3.1
zlib_src_dir=$(ospath ${source_dir}/zlib-${zlib_version})
zlib_build_dir=$(ospath ${build_dir}/zlib-${zlib_version}/)
zlib_install_dir=$(ospath ${install_dir}/zlib-${zlib_version}/)
zlib_tarball=$(ospath ${source_dir}/zlib-${zlib_version}.tar.gz)

# build only if install doesn't exist
if [ ! -d ${zlib_install_dir} ]; then
if ${build_zlib}; then
if [ ! -d ${zlib_src_dir} ]; then
  echo "**** Downloading ${zlib_tarball}"
  curl -L https://github.com/madler/zlib/releases/download/v${zlib_version}/zlib-${zlib_version}.tar.gz -o ${zlib_tarball}
  tar  ${tar_extra_args} -xzf ${zlib_tarball} -C ${source_dir}
fi

echo "**** Configuring Zlib ${zlib_version}"
cmake -S ${zlib_src_dir} -B ${zlib_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose} \
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DCMAKE_INSTALL_PREFIX=${zlib_install_dir}

echo "**** Building Zlib ${zlib_version}"
cmake --build ${zlib_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing Zlib ${zlib_version}"
cmake --install ${zlib_build_dir} --config ${build_config}

fi
else
  echo "**** Skipping Zlib build, install found at: ${zlib_install_dir}"
fi # build_zlib


################
# HDF5
################
# release 1-2 GAH!
hdf5_version=1.14.1-2
hdf5_middle_version=1.14.1
hdf5_short_version=1.14
hdf5_src_dir=$(ospath ${source_dir}/hdf5-${hdf5_version})
hdf5_build_dir=$(ospath ${build_dir}/hdf5-${hdf5_version}/)
hdf5_install_dir=$(ospath ${install_dir}/hdf5-${hdf5_version}/)
hdf5_tarball=$(ospath ${source_dir}/hdf5-${hdf5_version}.tar.gz)

# build only if install doesn't exist
if [ ! -d ${hdf5_install_dir} ]; then
if ${build_hdf5}; then
if [ ! -d ${hdf5_src_dir} ]; then
  echo "**** Downloading ${hdf5_tarball}"
  curl -L https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${hdf5_short_version}/hdf5-${hdf5_middle_version}/src/hdf5-${hdf5_version}.tar.gz -o ${hdf5_tarball}
  tar ${tar_extra_args} -xzf ${hdf5_tarball} -C ${source_dir}
fi

#################
#
# hdf5 1.14.x CMake recipe for using zlib
#
# -DHDF5_ENABLE_Z_LIB_SUPPORT=ON
# Add zlib install dir to CMAKE_PREFIX_PATH
#
#################

echo "**** Configuring HDF5 ${hdf5_version}"
cmake -S ${hdf5_src_dir} -B ${hdf5_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose} \
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DHDF5_ENABLE_Z_LIB_SUPPORT=ON \
  -DCMAKE_PREFIX_PATH=${zlib_install_dir} \
  -DCMAKE_INSTALL_PREFIX=${hdf5_install_dir}

echo "**** Building HDF5 ${hdf5_version}"
cmake --build ${hdf5_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing HDF5 ${hdf5_version}"
cmake --install ${hdf5_build_dir} --config ${build_config}

fi
else
  echo "**** Skipping HDF5 build, install found at: ${hdf5_install_dir}"
fi # build_hdf5

################
# Silo
################
silo_version=4.11.1
silo_src_dir=$(ospath ${source_dir}/silo-${silo_version})
silo_build_dir=$(ospath ${build_dir}/silo-${silo_version}/)
silo_install_dir=$(ospath ${install_dir}/silo-${silo_version}/)
silo_tarball=$(ospath ${source_dir}/silo-${silo_version}.tar.gz)

# build only if install doesn't exist
if [ ! -d ${silo_install_dir} ]; then
if ${build_silo}; then
if [ ! -d ${silo_src_dir} ]; then
  echo "**** Downloading ${silo_tarball}"
  curl -L https://github.com/LLNL/Silo/archive/refs/tags/${silo_version}.tar.gz -o ${silo_tarball}
  tar ${tar_extra_args} -xzf ${silo_tarball} -C ${source_dir}
  # apply silo patches
  cd  ${silop_src_dir}
  patch -p1 < ${script_dir}/2024_07_25_silo_4_11_cmake_fix.patch
  cd ${root_dir}
fi


echo "**** Configuring Silo ${silo_version}"
cmake -S ${silo_src_dir} -B ${silo_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose} \
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DCMAKE_INSTALL_PREFIX=${silo_install_dir} \
  -DSILO_ENABLE_SHARED=${build_shared_libs} \
  -DSILO_ENABLE_HDF5=ON \
  -DSILO_ENABLE_TESTS=OFF \
  -DSILO_BUILD_FOR_BSD_LICENSE=ON \
  -DSILO_ENABLE_FORTRAN=OFF \
  -DSILO_HDF5_DIR=${hdf5_install_dir}/cmake/ \
  -DCMAKE_PREFIX_PATH=${zlib_install_dir}


echo "**** Building Silo ${silo_version}"
cmake --build ${silo_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing Silo ${silo_version}"
cmake --install ${silo_build_dir} --config ${build_config}

fi
else
  echo "**** Skipping Silo build, install found at: ${silo_install_dir}"
fi # build_silo

############################
# Python Virtual Env
############################
python_exe=python3
venv_install_dir=$(ospath ${install_dir}/python-venv/)
venv_python_exe=$(ospath ${venv_install_dir}/bin/python3)
venv_sphinx_exe=$(ospath ${venv_install_dir}/bin/sphinx-build)

# build only if install doesn't exist
if [ ! -d ${venv_install_dir} ]; then
if ${build_pyvenv}; then
    echo "**** Creating Python Virtual Env"
    cd ${install_dir} && ${python_exe} -m venv python-venv
    ${venv_python_exe} -m pip install --upgrade pip
    ${venv_python_exe} -m pip install numpy sphinx sphinx_rtd_theme
    if [[ "$enable_mpi" == "ON" ]]; then
        ${venv_python_exe} -m pip install mpi4py
    fi
fi
else
  echo "**** Skipping Python venv build, install found at: ${venv_install_dir}"
fi # build_pyvenv

################
# Conduit
################
conduit_version=v0.9.2
conduit_src_dir=$(ospath ${source_dir}/conduit-${conduit_version}/src)
conduit_build_dir=$(ospath ${build_dir}/conduit-${conduit_version}/)
conduit_install_dir=$(ospath ${install_dir}/conduit-${conduit_version}/)
conduit_tarball=$(ospath ${source_dir}/conduit-${conduit_version}-src-with-blt.tar.gz)

# build only if install doesn't exist
if [ ! -d ${conduit_install_dir} ]; then
if ${build_conduit}; then
if [ ! -d ${conduit_src_dir} ]; then
  echo "**** Downloading ${conduit_tarball}"
  curl -L https://github.com/LLNL/conduit/releases/download/${conduit_version}/conduit-${conduit_version}-src-with-blt.tar.gz -o ${conduit_tarball}
  tar ${tar_extra_args} --exclude="conduit-${conduit_version}/src/tests/relay/data/silo/*" -x -v -f ${conduit_tarball} -C ${source_dir}
fi

#
# python settings
#
conduit_py_cmake_opts=-DENABLE_PYTHON=${enable_python}
if ${build_pyvenv}; then
  conduit_py_cmake_opts="${conduit_py_cmake_opts} -DPYTHON_EXECUTABLE=${venv_python_exe}"
  conduit_py_cmake_opts="${conduit_py_cmake_opts} -DSPHINX_EXECUTABLE=${venv_sphinx_exe}"
fi

echo "**** Configuring Conduit ${conduit_version}"
cmake -S ${conduit_src_dir} -B ${conduit_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose} \
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DCMAKE_INSTALL_PREFIX=${conduit_install_dir} \
  -DENABLE_FORTRAN=${enable_fortran} \
  -DENABLE_MPI=${enable_mpi} \
  -DENABLE_FIND_MPI=${enable_find_mpi} \
   ${conduit_py_cmake_opts} \
  -DENABLE_TESTS=${enable_tests} \
  -DHDF5_DIR=${hdf5_install_dir} \
  -DZLIB_DIR=${zlib_install_dir}


echo "**** Building Conduit ${conduit_version}"
cmake --build ${conduit_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing Conduit ${conduit_version}"
cmake --install ${conduit_build_dir} --config ${build_config}

fi
else
  echo "**** Skipping Conduit build, install found at: ${conduit_install_dir}"
fi # build_conduit

#########################
# Kokkos (only for hip)
#########################
kokkos_version=3.7.02
kokkos_src_dir=$(ospath ${source_dir}/kokkos-${kokkos_version})
kokkos_build_dir=$(ospath ${build_dir}kokkos-${kokkos_version})
kokkos_install_dir=$(ospath ${install_dir}/kokkos-${kokkos_version}/)
kokkos_tarball=$(ospath ${source_dir}/kokkos-${kokkos_version}.tar.gz)

if [[ "$enable_hip" == "ON" ]]; then
# build only if install doesn't exist
if [ ! -d ${kokkos_install_dir} ]; then
if ${build_kokkos}; then
if [ ! -d ${kokkos_src_dir} ]; then
  echo "**** Downloading ${kokkos_tarball}"
  curl -L https://github.com/kokkos/kokkos/archive/refs/tags/${kokkos_version}.tar.gz -o ${kokkos_tarball}
  tar ${tar_extra_args} -xzf ${kokkos_tarball} -C ${source_dir}
fi

# TODO: DKokkos_ARCH_VEGA90A needs to be controlled / mapped?

echo "**** Configuring Kokkos ${kokkos_version}"
cmake -S ${kokkos_src_dir} -B ${kokkos_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DKokkos_ARCH_VEGA90A=ON \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=OFF \
  -DCMAKE_INSTALL_PREFIX=${kokkos_install_dir} \
  -DCMAKE_CXX_FLAGS="--amdgpu-target=${ROCM_ARCH}" \
  -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX=${kokkos_install_dir}

echo "**** Building Kokkos ${kokkos_version}"
cmake --build ${kokkos_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing VTK-m ${kokkos_version}"
cmake --install ${kokkos_build_dir} --config ${build_config}

fi
else
  echo "**** Skipping Kokkos build, install found at: ${kokkos_install_dir}"
fi # build_kokkos

fi # if enable_hip

################
# VTK-m
################
vtkm_version=v2.1.0
vtkm_src_dir=$(ospath ${source_dir}/vtk-m-${vtkm_version})
vtkm_build_dir=$(ospath ${build_dir}/vtk-m-${vtkm_version})
vtkm_install_dir=$(ospath ${install_dir}/vtk-m-${vtkm_version}/)
vtkm_tarball=$(ospath ${source_dir}/vtk-m-${vtkm_version}.tar.gz)

# build only if install doesn't exist
if [ ! -d ${vtkm_install_dir} ]; then
if ${build_vtkm}; then
if [ ! -d ${vtkm_src_dir} ]; then
  echo "**** Downloading ${vtkm_tarball}"
  curl -L https://gitlab.kitware.com/vtk/vtk-m/-/archive/${vtkm_version}/vtk-m-${vtkm_version}.tar.gz -o ${vtkm_tarball}
  tar ${tar_extra_args} -xzf ${vtkm_tarball} -C ${source_dir}

  # apply vtk-m patch
  cd  ${vtkm_src_dir}
  patch -p1 < ${script_dir}/2023_12_06_vtkm-mr3160-rocthrust-fix.patch
  patch -p1 < ${script_dir}/2024_05_03_vtkm-mr3215-ext-geom-fix.patch
  patch -p1 < ${script_dir}/2024_07_02_vtkm-mr3246-raysubset_bugfix.patch
  cd ${root_dir}
fi


vtkm_extra_cmake_args=""
if [[ "$enable_cuda" == "ON" ]]; then
  vtkm_extra_cmake_args="-DVTKm_ENABLE_CUDA=ON"
  vtkm_extra_cmake_args="${vtkm_extra_cmake_args} -DCMAKE_CUDA_HOST_COMPILER=${CXX}"
  vtkm_extra_cmake_args="${vtkm_extra_cmake_args} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
fi

if [[ "$enable_hip" == "ON" ]]; then
  vtkm_extra_cmake_args="-DVTKm_ENABLE_KOKKOS=ON"
  vtkm_extra_cmake_args="${vtkm_extra_cmake_args} -DCMAKE_PREFIX_PATH=${kokkos_install_dir}"
  vtkm_extra_cmake_args="${vtkm_extra_cmake_args} -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH}"
  vtkm_extra_cmake_args="${vtkm_extra_cmake_args} -DVTKm_ENABLE_KOKKOS_THRUST=OFF"
fi

echo "**** Configuring VTK-m ${vtkm_version}"
cmake -S ${vtkm_src_dir} -B ${vtkm_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DVTKm_NO_DEPRECATED_VIRTUAL=ON \
  -DVTKm_USE_64BIT_IDS=OFF \
  -DVTKm_USE_DOUBLE_PRECISION=ON \
  -DVTKm_USE_DEFAULT_TYPES_FOR_ASCENT=ON \
  -DVTKm_ENABLE_MPI=${enable_mpi} \
  -DVTKm_ENABLE_OPENMP=${enable_openmp}\
  -DVTKm_ENABLE_RENDERING=ON \
  -DVTKm_ENABLE_TESTING=${enable_tests} \
  -DBUILD_TESTING=${enable_tests} \
  -DVTKm_ENABLE_BENCHMARKS=OFF ${vtkm_extra_cmake_args} \
  -DCMAKE_INSTALL_PREFIX=${vtkm_install_dir}

echo "**** Building VTK-m ${vtkm_version}"
cmake --build ${vtkm_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing VTK-m ${vtkm_version}"
cmake --install ${vtkm_build_dir}  --config ${build_config}

fi
else
  echo "**** Skipping VTK-m build, install found at: ${vtkm_install_dir}"
fi # build_vtkm


################
# Camp
################
camp_version=v2024.02.1
camp_src_dir=$(ospath ${source_dir}/camp-${camp_version})
camp_build_dir=$(ospath ${build_dir}/camp-${camp_version})
camp_install_dir=$(ospath ${install_dir}/camp-${camp_version}/)
camp_tarball=$(ospath ${source_dir}/camp-${camp_version}.tar.gz)


# build only if install doesn't exist
if [ ! -d ${camp_install_dir} ]; then
if ${build_camp}; then
if [ ! -d ${camp_src_dir} ]; then
  echo "**** Downloading ${camp_tarball}"
  curl -L https://github.com/LLNL/camp/releases/download/${camp_version}/camp-${camp_version}.tar.gz -o ${camp_tarball}
  tar ${tar_extra_args} -xzf ${camp_tarball} -C ${source_dir}
fi

camp_extra_cmake_args=""
if [[ "$enable_cuda" == "ON" ]]; then
  camp_extra_cmake_args="-DENABLE_CUDA=ON"
  camp_extra_cmake_args="${camp_extra_cmake_args} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
fi

if [[ "$enable_hip" == "ON" ]]; then
    camp_extra_cmake_args="-DENABLE_HIP=ON"
    camp_extra_cmake_args="${camp_extra_cmake_args} -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH}"
    camp_extra_cmake_args="${camp_extra_cmake_args} -DROCM_PATH=${ROCM_PATH}"
fi

echo "**** Configuring Camp ${camp_version}"
cmake -S ${camp_src_dir} -B ${camp_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DENABLE_TESTS=${enable_tests} \
  -DENABLE_EXAMPLES=${enable_tests} ${camp_extra_cmake_args} \
  -DCMAKE_INSTALL_PREFIX=${camp_install_dir}

echo "**** Building Camp ${camp_version}"
cmake --build ${camp_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing Camp ${camp_version}"
cmake --install ${camp_build_dir}  --config ${build_config}

fi
else
  echo "**** Skipping Camp build, install found at: ${camp_install_dir}"
fi # build_camp


################
# RAJA
################
raja_version=v2024.02.1
raja_src_dir=$(ospath ${source_dir}/RAJA-${raja_version})
raja_build_dir=$(ospath ${build_dir}/raja-${raja_version})
raja_install_dir=$(ospath ${install_dir}/raja-${raja_version}/)
raja_tarball=$(ospath ${source_dir}/RAJA-${raja_version}.tar.gz)
raja_enable_vectorization="${raja_enable_vectorization:=ON}"

# build only if install doesn't exist
if [ ! -d ${raja_install_dir} ]; then
if ${build_raja}; then
if [ ! -d ${raja_src_dir} ]; then
  echo "**** Downloading ${raja_tarball}"
  curl -L https://github.com/LLNL/RAJA/releases/download/${raja_version}/RAJA-${raja_version}.tar.gz -o ${raja_tarball}
  tar ${tar_extra_args} -xzf ${raja_tarball} -C ${source_dir}
fi

raja_extra_cmake_args=""
if [[ "$enable_cuda" == "ON" ]]; then
  raja_extra_cmake_args="-DENABLE_CUDA=ON"
  raja_extra_cmake_args="${raja_extra_cmake_args} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
fi

if [[ "$enable_hip" == "ON" ]]; then
  raja_extra_cmake_args="-DENABLE_HIP=ON"
  raja_extra_cmake_args="${raja_extra_cmake_args} -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH}"
  raja_extra_cmake_args="${raja_extra_cmake_args} -DROCM_PATH=${ROCM_PATH}"
fi

echo "**** Configuring RAJA ${raja_version}"
cmake -S ${raja_src_dir} -B ${raja_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -Dcamp_DIR=${camp_install_dir} \
  -DENABLE_OPENMP=${enable_openmp} \
  -DENABLE_TESTS=${enable_tests} \
  -DRAJA_ENABLE_TESTS=${enable_tests} \
  -DENABLE_EXAMPLES=${enable_tests} \
  -DENABLE_EXERCISES=${enable_tests} ${raja_extra_cmake_args} \
  -DCMAKE_INSTALL_PREFIX=${raja_install_dir} \
  -DRAJA_ENABLE_VECTORIZATION=${raja_enable_vectorization}

echo "**** Building RAJA ${raja_version}"
cmake --build ${raja_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing RAJA ${raja_version}"
cmake --install ${raja_build_dir}  --config ${build_config}

fi
else
  echo "**** Skipping RAJA build, install found at: ${raja_install_dir}"
fi # build_raja

################
# Umpire
################
umpire_version=2024.02.1
umpire_src_dir=$(ospath ${source_dir}/umpire-${umpire_version})
umpire_build_dir=$(ospath ${build_dir}/umpire-${umpire_version})
umpire_install_dir=$(ospath ${install_dir}/umpire-${umpire_version}/)
umpire_tarball=$(ospath ${source_dir}/umpire-${umpire_version}.tar.gz)
umpire_windows_cmake_flags="-DBLT_CXX_STD=c++17 -DCMAKE_CXX_STANDARD=17 -DUMPIRE_ENABLE_FILESYSTEM=On -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=On"

umpire_extra_cmake_args=""
if [[ "$build_windows" == "ON" ]]; then
  umpire_extra_cmake_args="${umpire_windows_cmake_flags}"
fi 

if [[ "$enable_cuda" == "ON" ]]; then
  umpire_extra_cmake_args="${umpire_extra_cmake_args} -DENABLE_CUDA=ON"
  umpire_extra_cmake_args="${umpire_extra_cmake_args} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
fi

if [[ "$enable_hip" == "ON" ]]; then
  umpire_extra_cmake_args="${umpire_extra_cmake_args} -DENABLE_HIP=ON"
  umpire_extra_cmake_args="${umpire_extra_cmake_args} -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH}"
  umpire_extra_cmake_args="${umpire_extra_cmake_args} -DROCM_PATH=${ROCM_PATH}"
fi

# build only if install doesn't exist
if [ ! -d ${umpire_install_dir} ]; then
if ${build_umpire}; then
if [ ! -d ${umpire_src_dir} ]; then
  echo "**** Downloading ${umpire_tarball}"
  curl -L https://github.com/LLNL/Umpire/releases/download/v${umpire_version}/umpire-${umpire_version}.tar.gz -o ${umpire_tarball}
  tar ${tar_extra_args} -xzf ${umpire_tarball} -C ${source_dir}
fi

echo "**** Configuring Umpire ${umpire_version}"
cmake -S ${umpire_src_dir} -B ${umpire_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose} \
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -Dcamp_DIR=${camp_install_dir} \
  -DENABLE_OPENMP=${enable_openmp} \
  -DENABLE_TESTS=${enable_tests} \
  -DUMPIRE_ENABLE_TOOLS=Off \
  -DUMPIRE_ENABLE_BENCHMARKS=${enable_tests} ${umpire_extra_cmake_args} \
  -DCMAKE_INSTALL_PREFIX=${umpire_install_dir}

echo "**** Building Umpire ${umpire_version}"
cmake --build ${umpire_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing Umpire ${umpire_version}"
cmake --install ${umpire_build_dir}  --config ${build_config}

fi
else
  echo "**** Skipping Umpire build, install found at: ${umpire_install_dir}"
fi # build_umpire

################
# MFEM
################
mfem_version=4.6
mfem_src_dir=$(ospath ${source_dir}/mfem-${mfem_version})
mfem_build_dir=$(ospath ${build_dir}/mfem-${mfem_version})
mfem_install_dir=$(ospath ${install_dir}/mfem-${mfem_version}/)
mfem_tarball=$(ospath ${source_dir}/mfem-${mfem_version}.tar.gz)
mfem_windows_cmake_flags="-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON"

mfem_extra_cmake_args=""
if [[ "$build_windows" == "ON" ]]; then
  mfem_extra_cmake_args="${mfem_windows_cmake_flags}"
fi 


# build only if install doesn't exist
if [ ! -d ${mfem_install_dir} ]; then
if ${build_mfem}; then
if [ ! -d ${mfem_src_dir} ]; then
  echo "**** Downloading ${mfem_tarball}"
  curl -L https://github.com/mfem/mfem/archive/refs/tags/v${mfem_version}.tar.gz -o ${mfem_tarball}
  tar ${tar_extra_args} -xzf ${mfem_tarball} -C ${source_dir}
fi


echo "**** Configuring MFEM ${mfem_version}"
cmake -S ${mfem_src_dir} -B ${mfem_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DMFEM_USE_CONDUIT=ON ${mfem_extra_cmake_args} \
  -DCMAKE_PREFIX_PATH="${conduit_install_dir}" \
  -DMFEM_ENABLE_TESTING=${enable_tests} \
  -DMFEM_ENABLE_EXAMPLES=${enable_tests} \
  -DCMAKE_INSTALL_PREFIX=${mfem_install_dir} 

echo "**** Building MFEM ${mfem_version}"
cmake --build ${mfem_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing MFEM ${mfem_version}"
cmake --install ${mfem_build_dir}  --config ${build_config}

fi
else
  echo "**** Skipping MFEM build, install found at: ${mfem_install_dir}"
fi # build_mfem

################
# Catalyst
################
catalyst_version=2.0.0-rc4
catalyst_src_dir=$(ospath ${source_dir}/catalyst-v${catalyst_version})
catalyst_build_dir=$(ospath ${build_dir}/catalyst-v${catalyst_version})
catalyst_install_dir=$(ospath ${install_dir}/catalyst-v${catalyst_version}/)
catalyst_cmake_dir=${catalyst_install_dir}lib64/cmake/catalyst-2.0/
catalyst_tarball=$(ospath ${source_dir}/catalyst-v${catalyst_version}.tar.gz)

# build only if install doesn't exist
if [ ! -d ${catalyst_install_dir} ]; then
if ${build_catalyst}; then
if [ ! -d ${catalyst_src_dir} ]; then
  echo "**** Downloading ${catalyst_tarball}"
  curl -L https://gitlab.kitware.com/paraview/catalyst/-/archive/v${catalyst_version}/catalyst-v${catalyst_version}.tar.gz -o ${catalyst_tarball}
  tar ${tar_extra_args} -xzf ${catalyst_tarball} -C ${source_dir}
fi

echo "**** Configuring Catalyst ${catalyst_version}"
cmake -S ${catalyst_src_dir} -B ${catalyst_build_dir} ${cmake_compiler_settings} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DCATALYST_BUILD_TESTING=${enable_tests} \
  -DCATALYST_USE_MPI=${enable_mpi} \
  -DCMAKE_INSTALL_PREFIX=${catalyst_install_dir} \

echo "**** Building Catalyst ${catalyst_version}"
cmake --build ${catalyst_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing Catalyst ${catalyst_version}"
cmake --install ${catalyst_build_dir} --config ${build_config}

fi
else
  echo "**** Skipping Catalyst build, install found at: ${catalyst_install_dir}"
fi # build_catalyst

################
# Ascent
################
# if we are in an ascent checkout, use existing source
ascent_checkout_dir=$(ospath ${script_dir}/../../src)
ascent_checkout_dir=$(abs_path ${ascent_checkout_dir})
echo ${ascent_checkout_dir}
if [ -d ${ascent_checkout_dir} ]; then
    ascent_version=checkout
    ascent_src_dir=$(abs_path ${ascent_checkout_dir})
    echo "**** Using existing Ascent source repo checkout: ${ascent_src_dir}"
else
    ascent_version=develop
    ascent_src_dir=$(ospath ${source_dir}/ascent/src)
fi

# otherwise use ascent develop
ascent_build_dir=$(ospath ${build_dir}/ascent-${ascent_version}/)
ascent_install_dir=$(ospath ${install_dir}//ascent-${ascent_version}/)

echo "**** Creating Ascent host-config (ascent-config.cmake)"
#
echo '# host-config file generated by build_ascent.sh' > ${root_dir}/ascent-config.cmake

# capture compilers if they are provided via env vars
if [ ! -z ${CC+x} ]; then
    echo 'set(CMAKE_C_COMPILER ' ${CC} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi

if [ ! -z ${CXX+x} ]; then
    echo 'set(CMAKE_CXX_COMPILER ' ${CXX} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi

if [ ! -z ${FTN+x} ]; then
    echo 'set(CMAKE_Fortran_COMPILER ' ${FTN} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi

# capture compiler flags  if they are provided via env vars
if [ ! -z ${CFLAGS+x} ]; then
    echo 'set(CMAKE_C_FLAGS "' ${CFLAGS} '" CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi

if [ ! -z ${CXXFLAGS+x} ]; then
    echo 'set(CMAKE_CXX_FLAGS "' ${CXXFLAGS} '" CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi

if [ ! -z ${FFLAGS+x} ]; then
    echo 'set(CMAKE_F_FLAGS "' ${FFLAGS} '" CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi

echo 'set(CMAKE_VERBOSE_MAKEFILE ' ${enable_verbose} ' CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
echo 'set(CMAKE_BUILD_TYPE ' ${build_config} ' CACHE STRING "")' >> ${root_dir}/ascent-config.cmake
echo 'set(BUILD_SHARED_LIBS ' ${build_shared_libs} ' CACHE STRING "")' >> ${root_dir}/ascent-config.cmake
echo 'set(CMAKE_INSTALL_PREFIX ' ${ascent_install_dir} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
echo 'set(ENABLE_TESTS ' ${enable_tests} ' CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
echo 'set(ENABLE_MPI ' ${enable_mpi} ' CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
echo 'set(ENABLE_FIND_MPI ' ${enable_find_mpi} ' CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
echo 'set(ENABLE_FORTRAN ' ${enable_fortran} ' CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
echo 'set(ENABLE_PYTHON ' ${enable_python} ' CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
if ${build_pyvenv}; then
echo 'set(PYTHON_EXECUTABLE ' ${venv_python_exe} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
echo 'set(ENABLE_DOCS ON CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
echo 'set(SPHINX_EXECUTABLE ' ${venv_sphinx_exe} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi
echo 'set(BLT_CXX_STD c++14 CACHE STRING "")' >> ${root_dir}/ascent-config.cmake
echo 'set(CONDUIT_DIR ' ${conduit_install_dir} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
echo 'set(VTKM_DIR ' ${vtkm_install_dir} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
echo 'set(CAMP_DIR ' ${camp_install_dir} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
echo 'set(RAJA_DIR ' ${raja_install_dir} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
echo 'set(UMPIRE_DIR ' ${umpire_install_dir} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
echo 'set(MFEM_DIR ' ${mfem_install_dir} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
echo 'set(ENABLE_VTKH ON CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
echo 'set(ENABLE_APCOMP ON CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
echo 'set(ENABLE_DRAY ON CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake


if ${build_catalyst}; then
    echo 'set(CATALYST_DIR ' ${catalyst_cmake_dir} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi

if [[ "$enable_cuda" == "ON" ]]; then
    echo 'set(ENABLE_CUDA ON CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
    echo 'set(CMAKE_CUDA_ARCHITECTURES ' ${CUDA_ARCH} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi

if [[ "$enable_hip" == "ON" ]]; then
    echo 'set(ENABLE_HIP ON CACHE BOOL "")' >> ${root_dir}/ascent-config.cmake
    echo 'set(CMAKE_HIP_ARCHITECTURES ' ${ROCM_ARCH} ' CACHE STRING "")' >> ${root_dir}/ascent-config.cmake
    echo 'set(ROCM_PATH ' ${ROCM_PATH} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
    echo 'set(KOKKOS_DIR ' ${kokkos_install_dir} ' CACHE PATH "")' >> ${root_dir}/ascent-config.cmake
fi

# build only if install doesn't exist
if [ ! -d ${ascent_install_dir} ]; then
if ${build_ascent}; then
if [ ! -d ${ascent_src_dir} ]; then
    echo "**** Cloning Ascent"
    git clone --recursive https://github.com/Alpine-DAV/ascent.git
fi

echo "**** Configuring Ascent"
cmake -S ${ascent_src_dir} -B ${ascent_build_dir} -C ${root_dir}/ascent-config.cmake

echo "**** Building Ascent"
cmake --build ${ascent_build_dir} --config ${build_config} -j${build_jobs}

echo "**** Installing Ascent"
cmake --install ${ascent_build_dir}  --config ${build_config}

if ${build_catalyst}; then
    mv ${ascent_install_dir}/lib/libcatalyst-ascent.so ${catalyst_install_dir}lib64/catalyst/libcatalyst-ascent.so
fi

fi
else
  echo "**** Skipping Ascent build, install found at: ${ascent_install_dir}"
fi # build_ascent
