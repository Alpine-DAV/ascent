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
#  - selected compilers (including nvcc) are in your path or set via env vars
#  - [when enabled] MPI and Python (+numpy and mpi4py), are in your path
#
##############################################################################
set -eu -o pipefail

CC="${CC:=gcc}"
CXX="${CXX:=g++}"
FTN="${FTN:=gfortran}"

CUDA_ARCH="${CUDA_ARCH:=80}"
CUDA_ARCH_VTKM="${CUDA_ARCH_VTKM:=ampere}"

##############################################################################
# Build Options
##############################################################################

# shared options
enable_fortran="${enable_fortran:=OFF}"
enable_python="${enable_python:=OFF}"
enable_openmp="${enable_openmp:=OFF}"
enable_mpi="${enable_mpi:=OFF}"
enable_find_mpi="${enable_find_mpi:=ON}"
enable_tests="${enable_tests:=ON}"
enable_verbose="${enable_verbose:=ON}"
build_jobs="${build_jobs:=6}"
build_config="${build_config:=Release}"
build_shared_libs="${build_shared_libs:=ON}"

# tpl controls
build_hdf5="${build_hdf5:=true}"
build_conduit="${build_conduit:=true}"
build_vtkm="${build_vtkm:=true}"
build_camp="${build_camp:=true}"
build_raja="${build_raja:=true}"
build_umpire="${build_umpire:=true}"
build_mfem="${build_mfem:=true}"

# ascent options
build_ascent="${build_ascent:=true}"

root_dir=$(pwd)

################
# HDF5
################
hdf5_version=1.12.2
hdf5_src_dir=${root_dir}/hdf5-${hdf5_version}
hdf5_build_dir=${root_dir}/build/hdf5-${hdf5_version}/
hdf5_install_dir=${root_dir}/install/hdf5-${hdf5_version}/
hdf5_tarball=hdf5-${hdf5_version}.tar.gz

# build only if install doesn't exist
if [ ! -d ${hdf5_install_dir} ]; then
if ${build_hdf5}; then
if [ ! -d ${hdf5_src_dir} ]; then
  echo "**** Downloading ${hdf5_tarball}"
  curl -L https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.gz -o ${hdf5_tarball}
  tar -xzf ${hdf5_tarball}
fi

echo "**** Configuring HDF5 ${hdf5_version}"
cmake -S ${hdf5_src_dir} -B ${hdf5_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose} \
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DCMAKE_INSTALL_PREFIX=${hdf5_install_dir}

echo "**** Building HDF5 ${hdf5_version}"
cmake --build ${hdf5_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing HDF5 ${hdf5_version}"
cmake --install ${hdf5_build_dir}

fi
else
  echo "**** Skipping HDF5 build, install found at: ${hdf5_install_dir}"
fi # build_hdf5


################
# Conduit
################
conduit_version=v0.8.6
conduit_src_dir=${root_dir}/conduit-${conduit_version}/src
conduit_build_dir=${root_dir}/build/conduit-${conduit_version}/
conduit_install_dir=${root_dir}/install/conduit-${conduit_version}/
conduit_tarball=conduit-${conduit_version}-src-with-blt.tar.gz

# build only if install doesn't exist
if [ ! -d ${conduit_install_dir} ]; then
if ${build_conduit}; then
if [ ! -d ${conduit_src_dir} ]; then
  echo "**** Downloading ${conduit_tarball}"
  curl -L https://github.com/LLNL/conduit/releases/download/${conduit_version}/${conduit_tarball} -o ${conduit_tarball}
  tar -xzf ${conduit_tarball}
fi

echo "**** Configuring Conduit ${conduit_version}"
cmake -S ${conduit_src_dir} -B ${conduit_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DCMAKE_INSTALL_PREFIX=${conduit_install_dir} \
  -DENABLE_FORTRAN=${enable_fortran} \
  -DENABLE_MPI=${enable_mpi} \
  -DENABLE_FIND_MPI=${enable_find_mpi} \
  -DENABLE_PYTHON=${enable_python} \
  -DENABLE_TESTS=${enable_tests} \
  -DHDF5_DIR=${hdf5_install_dir}

echo "**** Building Conduit ${conduit_version}"
cmake --build ${conduit_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing Conduit ${conduit_version}"
cmake --install ${conduit_build_dir}

fi
else
  echo "**** Skipping Conduit build, install found at: ${conduit_install_dir}"
fi # build_conduit


################
# VTK-m
################
vtkm_version=v1.9.0
vtkm_src_dir=${root_dir}/vtk-m-${vtkm_version}
vtkm_build_dir=${root_dir}/build/vtk-m-${vtkm_version}
vtkm_install_dir=${root_dir}/install/vtk-m-${vtkm_version}/
vtkm_tarball=vtk-m-${vtkm_version}.tar.gz

# build only if install doesn't exist
if [ ! -d ${vtkm_install_dir} ]; then
if ${build_vtkm}; then
if [ ! -d ${vtkm_src_dir} ]; then
  echo "**** Downloading ${vtkm_tarball}"
  curl -L https://gitlab.kitware.com/vtk/vtk-m/-/archive/${vtkm_version}/${vtkm_tarball} -o ${vtkm_tarball}
  tar -xzf ${vtkm_tarball}
fi

echo "**** Configuring VTK-m ${vtkm_version}"
cmake -S ${vtkm_src_dir} -B ${vtkm_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DVTKm_NO_DEPRECATED_VIRTUAL=ON \
  -DVTKm_USE_64BIT_IDS=OFF \
  -DVTKm_USE_DOUBLE_PRECISION=ON \
  -DVTKm_USE_DEFAULT_TYPES_FOR_ASCENT=ON \
  -DVTKm_ENABLE_BENCHMARKS=OFF\
  -DVTKm_ENABLE_RENDERING=ON \
  -DVTKm_ENABLE_TESTING=OFF \
  -DBUILD_TESTING=OFF \
  -DVTKm_ENABLE_BENCHMARKS=OFF\
  -DVTKm_ENABLE_MPI=OFF \
  -DVTKm_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_HOST_COMPILER=${CXX}\
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
  -DCMAKE_INSTALL_PREFIX=${vtkm_install_dir}

echo "**** Building VTK-m ${vtkm_version}"
cmake --build ${vtkm_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing VTK-m ${vtkm_version}"
cmake --install ${vtkm_build_dir}

fi
else
  echo "**** Skipping VTK-m build, install found at: ${vtkm_install_dir}"
fi # build_vtkm


################
# Camp
################
camp_version=2022.10.1
camp_src_dir=${root_dir}/camp-${camp_version}
camp_build_dir=${root_dir}/build/camp-${camp_version}
camp_install_dir=${root_dir}/install/camp-${camp_version}/
camp_tarball=camp-${camp_version}.tar.gz

# build only if install doesn't exist
if [ ! -d ${camp_install_dir} ]; then
if ${build_camp}; then
if [ ! -d ${camp_src_dir} ]; then
  echo "**** Cloning Camp ${camp_version}"
  # clone since camp releases don't contain submodules
  git clone --recursive --depth 1 --branch v${camp_version} https://github.com/LLNL/camp.git camp-${camp_version}
  # curl -L https://github.com/LLNL/camp/archive/refs/tags/v${camp_version}.tar.gz -o ${camp_tarball} 
  # tar -xzf ${camp_tarball} 
fi

echo "**** Configuring Camp ${camp_version}"
cmake -S ${camp_src_dir} -B ${camp_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose} \
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
  -DENABLE_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=${camp_install_dir}

echo "**** Building Camp ${camp_version}"
cmake --build ${camp_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing Camp ${camp_version}"
cmake --install ${camp_build_dir}

fi
else
  echo "**** Skipping Camp build, install found at: ${camp_install_dir}"
fi # build_camp


################
# RAJA
################
raja_version=v2022.10.4
raja_src_dir=${root_dir}/RAJA-${raja_version}
raja_build_dir=${root_dir}/build/raja-${raja_version}
raja_install_dir=${root_dir}/install/raja-${raja_version}/
raja_tarball=RAJA-${raja_version}.tar.gz

# build only if install doesn't exist
if [ ! -d ${raja_install_dir} ]; then
if ${build_raja}; then
if [ ! -d ${raja_src_dir} ]; then
  echo "**** Downloading ${raja_tarball}"
  curl -L https://github.com/LLNL/RAJA/releases/download/${raja_version}/${raja_tarball} -o ${raja_tarball} 
  tar -xzf ${raja_tarball} 
fi

echo "**** Configuring RAJA ${raja_version}"
cmake -S ${raja_src_dir} -B ${raja_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DENABLE_OPENMP=OFF \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
  -Dcamp_DIR=${camp_install_dir} \
  -DENABLE_TESTS=${enable_tests} \
  -DRAJA_ENABLE_TESTS=${enable_tests} \
  -DENABLE_EXAMPLES=${enable_tests} \
  -DENABLE_EXERCISES=${enable_tests} \
  -DRAJA_ENABLE_VECTORIZATION=OFF \
  -DCMAKE_INSTALL_PREFIX=${raja_install_dir}
  

echo "**** Building RAJA ${raja_version}"
cmake --build ${raja_build_dir} --config ${build_config} -j${build_jobs}
echo "**** Installing RAJA ${raja_version}"
cmake --install ${raja_build_dir}

fi
else
  echo "**** Skipping RAJA build, install found at: ${raja_install_dir}"
fi # build_raja


################
# Umpire
################
umpire_version=2022.10.0
umpire_src_dir=${root_dir}/umpire-${umpire_version}
umpire_build_dir=${root_dir}/build/umpire-${umpire_version}
umpire_install_dir=${root_dir}/install/umpire-${umpire_version}/
umpire_tarball=umpire-${umpire_version}.tar.gz

# build only if install doesn't exist
if [ ! -d ${umpire_install_dir} ]; then
if ${build_umpire}; then
if [ ! -d ${umpire_src_dir} ]; then
  echo "**** Downloading ${umpire_tarball}"
  curl -L https://github.com/LLNL/Umpire/releases/download/v${umpire_version}/${umpire_tarball} -o ${umpire_tarball}
  tar -xzf ${umpire_tarball}
fi

echo "**** Configuring Umpire ${umpire_version}"
cmake -S ${umpire_src_dir} -B ${umpire_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose} \
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
  -Dcamp_DIR=${camp_install_dir} \
  -DENABLE_OPENMP=${enable_openmp} \
  -DENABLE_TESTS=${enable_tests} \
  -DCMAKE_INSTALL_PREFIX=${umpire_install_dir}

echo "**** Building Umpire ${umpire_version}"
cmake --build ${umpire_build_dir} -j${build_jobs}
echo "**** Installing Umpire ${umpire_version}"
cmake --install ${umpire_build_dir}

fi
else
  echo "**** Skipping Umpire build, install found at: ${umpire_install_dir}"
fi # build_umpire

################
# MFEM
################
mfem_version=4.4
mfem_src_dir=${root_dir}/mfem-${mfem_version}
mfem_build_dir=${root_dir}/build/mfem-${mfem_version}
mfem_install_dir=${root_dir}/install/mfem-${mfem_version}/
mfem_tarball=mfem-${mfem_version}.tar.gz

# build only if install doesn't exist
if [ ! -d ${mfem_install_dir} ]; then
if ${build_mfem}; then
if [ ! -d ${mfem_src_dir} ]; then
  echo "**** Downloading ${mfem_tarball}"
  curl -L https://github.com/mfem/mfem/archive/refs/tags/v4.4.tar.gz -o ${mfem_tarball}
  tar -xzf ${mfem_tarball}
fi

echo "**** Configuring MFEM ${mfem_version}"
cmake -S ${mfem_src_dir} -B ${mfem_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose}\
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DMFEM_USE_CONDUIT=ON \
  -DCMAKE_PREFIX_PATH="${conduit_install_dir}" \
  -DCMAKE_INSTALL_PREFIX=${mfem_install_dir}

echo "**** Building MFEM ${vtkm_version}"
cmake --build ${mfem_build_dir} -j${build_jobs}
echo "**** Installing MFEM ${mfem_version}"
cmake --install ${mfem_build_dir}

fi
else
  echo "**** Skipping MFEM build, install found at: ${mfem_install_dir}"
fi # build_mfem


################
# Ascent
################
ascent_version=develop
ascent_src_dir=${root_dir}/ascent/src
ascent_build_dir=${root_dir}/build/ascent-${ascent_version}/
ascent_install_dir=${root_dir}/install/ascent-${ascent_version}/

echo "**** Creating Ascent host-config (ascent-config.cmake)"
#
echo '# host-config file generated by build_ascent.sh' > ascent-config.cmake
echo 'set(CMAKE_VERBOSE_MAKEFILE ' ${enable_verbose} 'CACHE BOOL "")' >> ascent-config.cmake
echo 'set(CMAKE_C_COMPILER ' ${CC} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(CMAKE_CXX_COMPILER ' ${CXX} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(CMAKE_BUILD_TYPE ' ${build_config} ' CACHE STRING "")' >> ascent-config.cmake
echo 'set(BUILD_SHARED_LIBS ' ${build_shared_libs} ' CACHE STRING "")' >> ascent-config.cmake
echo 'set(CMAKE_INSTALL_PREFIX ' ${ascent_install_dir} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(ENABLE_TESTS ' ${enable_tests} ' CACHE BOOL "")' >> ascent-config.cmake
echo 'set(ENABLE_MPI ' ${enable_mpi} ' CACHE BOOL "")' >> ascent-config.cmake
echo 'set(ENABLE_FIND_MPI ' ${enable_find_mpi} ' CACHE BOOL "")' >> ascent-config.cmake
echo 'set(ENABLE_FORTRAN ' ${enable_fortran} ' CACHE BOOL "")' >> ascent-config.cmake
echo 'set(ENABLE_PYTHON ' ${enable_python} ' CACHE BOOL "")' >> ascent-config.cmake
echo 'set(BLT_CXX_STD c++14 CACHE STRING "")' >> ascent-config.cmake
echo 'set(ENABLE_CUDA ON CACHE BOOL "")' >> ascent-config.cmake
echo 'set(CMAKE_CUDA_ARCHITECTURES ' ${CUDA_ARCH} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(CONDUIT_DIR ' ${conduit_install_dir} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(VTKM_DIR ' ${vtkm_install_dir} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(CAMP_DIR ' ${camp_install_dir} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(RAJA_DIR ' ${raja_install_dir} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(UMPIRE_DIR ' ${umpire_install_dir} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(MFEM_DIR ' ${mfem_install_dir} ' CACHE PATH "")' >> ascent-config.cmake
echo 'set(ENABLE_VTKH ON CACHE BOOL "")' >> ascent-config.cmake
echo 'set(ENABLE_APCOMP ON CACHE BOOL "")' >> ascent-config.cmake
echo 'set(ENABLE_DRAY ON CACHE BOOL "")' >> ascent-config.cmake

# build only if install doesn't exist
if [ ! -d ${ascent_install_dir} ]; then
if ${build_ascent}; then
if [ ! -d ${ascent_src_dir} ]; then
    echo "**** Cloning Ascent"
    git clone --recursive https://github.com/Alpine-DAV/ascent.git
fi

echo "**** Configuring Ascent"
cmake -S ${ascent_src_dir} -B ${ascent_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=${enable_verbose} \
  -DCMAKE_BUILD_TYPE=${build_config} \
  -DBUILD_SHARED_LIBS=${build_shared_libs} \
  -DCMAKE_INSTALL_PREFIX=${ascent_install_dir} \
  -DENABLE_MPI=${enable_mpi} \
  -DENABLE_FIND_MPI=${enable_find_mpi} \
  -DENABLE_FORTRAN=${enable_fortran} \
  -DENABLE_TESTS=$enable_tests \
  -DENABLE_PYTHON=${enable_python} \
  -DBLT_CXX_STD=c++14 \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
  -DCONDUIT_DIR=${conduit_install_dir} \
  -DVTKM_DIR=${vtkm_install_dir} \
  -DRAJA_DIR=${raja_install_dir} \
  -DUMPIRE_DIR=${umpire_install_dir} \
  -DCAMP_DIR=${camp_install_dir} \
  -DMFEM_DIR=${mfem_install_dir} \
  -DENABLE_VTKH=ON \
  -DENABLE_APCOMP=ON \
  -DENABLE_DRAY=ON

echo "**** Building Ascent"
cmake --build ${ascent_build_dir} -j${build_jobs}
echo "**** Installing Ascent"
cmake --install ${ascent_build_dir}

fi
else
  echo "**** Skipping Ascent build, install found at: ${ascent_install_dir}"
fi # build_ascent

