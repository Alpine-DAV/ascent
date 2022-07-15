#!/bin/bash

##############################################################################
# Demonstrates how to manually build Ascent and its dependencies.
#
#
# Assumes: `cmake` is in your path
#           Selected compilers are in path or env vars
##############################################################################
set -eu -o pipefail

##############################################################################
# Build Options
##############################################################################

# shared options
enable_fortran="${enable_fortran:=OFF}"
enable_python="${enable_python:=OFF}"
enable_openmp="${enable_openmp:=OFF}"
enable_mpi="${enable_mpi:=OFF}"
enable_tests="${enable_tests:=OFF}"

# tpls
build_hdf5="${build_hdf5:=true}"
build_conduit="${build_conduit:=true}"
build_vtkm="${build_vtkm:=true}"
build_raja="${build_raja:=true}"
build_umpire="${build_umpire:=true}"

# ascent options
build_ascent="${build_camp:=true}"

build_jobs="${build_jobs:=6}"

# HARDIRE
build_hdf5=false
build_conduit=false
build_vtkm=false
build_raja=true
build_umpire=false
build_ascent=true


root_dir=$(pwd)

################
# HDF5
################
hdf5_version=1.12.2
hdf5_src_dir=${root_dir}/hdf5-${hdf5_version}
hdf5_build_dir=${root_dir}/build/hdf5-${hdf5_version}/
hdf5_install_dir=${root_dir}/install/hdf5-${hdf5_version}/
hdf5_tarball=hdf5-${hdf5_version}.tar.gz

if ${build_hdf5}; then
if [ ! -d ${hdf5_src_dir} ]; then
  echo "**** Downloading ${hdf5_tarball}"
  curl -L https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.gz -o ${hdf5_tarball}
  tar -xzf ${hdf5_tarball}
fi

echo "**** Configuring HDF5 ${hdf5_version}"
cmake -S ${hdf5_src_dir} -B ${hdf5_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${hdf5_install_dir}

echo "**** Building HDF5 ${hdf5_version}"
cmake --build ${hdf5_build_dir} -j${build_jobs}
echo "**** Installing HDF5 ${hdf5_version}"
cmake --install ${hdf5_build_dir}

fi # build_hdf5

################
# Conduit
################
conduit_version=v0.8.3
conduit_src_dir=${root_dir}/conduit-${conduit_version}/src
conduit_build_dir=${root_dir}/build/conduit-${conduit_version}/
conduit_install_dir=${root_dir}/install/conduit-${conduit_version}/
conduit_tarball=conduit-${conduit_version}-src-with-blt.tar.gz

if ${build_conduit}; then
if [ ! -d ${conduit_src_dir} ]; then
  echo "**** Downloading ${conduit_tarball}"
  curl -L https://github.com/LLNL/conduit/releases/download/${conduit_version}/${conduit_tarball} -o ${conduit_tarball}
  tar -xzf ${conduit_tarball}
fi

echo "**** Configuring Conduit ${conduit_version}"
cmake -S ${conduit_src_dir} -B ${conduit_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${conduit_install_dir} \
  -DENABLE_FORTRAN=${enable_fortran} \
  -DENABLE_MPI=${enable_mpi} \
  -DENABLE_PYTHON=${enable_python} \
  -DENABLE_TESTS=${enable_tests} \
  -DHDF5_DIR=${hdf5_install_dir}

echo "**** Building Conduit ${conduit_version}"
cmake --build ${conduit_build_dir} -j${build_jobs}
echo "**** Installing Conduit ${conduit_version}"
cmake --install ${conduit_build_dir}

fi # build_conduit

################
# VTK-m
################
vtkm_version=v1.8.0
vtkm_src_dir=${root_dir}/vtk-m-${vtkm_version}
vtkm_build_dir=${root_dir}/build/vtk-m-${vtkm_version}
vtkm_install_dir=${root_dir}/install/vtk-m-${vtkm_version}/
vtkm_tarball=vtk-m-${vtkm_version}.tar.gz

if ${build_vtkm}; then
if [ ! -d ${vtkm_src_dir} ]; then
  echo "**** Downloading ${vtkm_tarball}"
  curl -L https://gitlab.kitware.com/vtk/vtk-m/-/archive/${vtkm_version}/${vtkm_tarball} -o ${vtkm_tarball}
  tar -xzf ${vtkm_tarball}
fi

echo "**** Configuring VTK-m ${vtkm_version}"
cmake -S ${vtkm_src_dir} -B ${vtkm_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF\
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON\
  -DVTKm_NO_DEPRECATED_VIRTUAL=ON \
  -DVTKm_USE_64BIT_IDS=OFF \
  -DVTKm_USE_DOUBLE_PRECISION=ON \
  -DVTKm_USE_DEFAULT_TYPES_FOR_ASCENT=ON \
  -DVTKm_ENABLE_MPI=OFF \
  -DVTKm_ENABLE_RENDERING=ON \
  -DVTKm_ENABLE_TESTING=OFF \
  -DBUILD_TESTING=OFF \
  -DVTKm_ENABLE_BENCHMARKS=OFF\
  -DCMAKE_INSTALL_PREFIX=${vtkm_install_dir}

echo "**** Building VTK-m ${vtkm_version}"
cmake --build ${vtkm_build_dir} -j6
echo "**** Installing VTK-m ${vtkm_version}"
cmake --install ${vtkm_build_dir}

fi # build_vtkm


################
# RAJA
################
raja_version=v0.14.1
raja_src_dir=${root_dir}/RAJA-${raja_version}
raja_build_dir=${root_dir}/build/raja-${raja_version}
raja_install_dir=${root_dir}/install/raja-${raja_version}/
raja_tarball=RAJA-${raja_version}.tar.gz

if ${build_raja}; then
if [ ! -d ${raja_src_dir} ]; then
  echo "**** Downloading ${raja_tarball}"
  curl -L https://github.com/LLNL/RAJA/releases/download/${raja_version}/${raja_tarball} -o ${raja_tarball} 
  tar -xzf ${raja_tarball} 
fi

echo "**** Configuring RAJA ${raja_version}"
cmake -S ${raja_src_dir} -B ${raja_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF\
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON\
  -DENABLE_OPENMP=${enable_openmp} \
  -DENABLE_TESTS=${enable_tests} \
  -DCMAKE_INSTALL_PREFIX=${raja_install_dir}

echo "**** Building RAJA ${raja_version}"
cmake --build ${raja_build_dir} -j6
echo "**** Installing RAJA ${raja_version}"
cmake --install ${raja_build_dir}

fi # build_raja

################
# Umpire
################
umpire_version=6.0.0
umpire_src_dir=${root_dir}/Umpire-${umpire_version}
umpire_build_dir=${root_dir}/build/umpire-${umpire_version}
umpire_install_dir=${root_dir}/install/umpire-${umpire_version}/
umpire_tarball=umpire-${umpire_version}.tar.gz

if ${build_umpire}; then
if [ ! -d ${umpire_src_dir} ]; then
  echo "**** Downloading ${umpire_tarball}"
  curl -L https://github.com/LLNL/Umpire/releases/download/v${umpire_version}/${umpire_tarball} -o ${umpire_tarball}
  tar -xzf ${umpire_tarball}
fi

echo "**** Configuring Umpire ${umpire_version}"
cmake -S ${umpire_src_dir} -B ${umpire_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF\
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DENABLE_OPENMP=${enable_openmp} \
  -DENABLE_TESTS=${enable_tests} \
  -DCMAKE_INSTALL_PREFIX=${umpire_install_dir}

echo "**** Building Umpire ${umpire_version}"
cmake --build ${umpire_build_dir} -j6
echo "**** Installing Umpire ${umpire_version}"
cmake --install ${umpire_build_dir}

fi # build_umpire


################
# Ascent
################
ascent_version=develop
ascent_src_dir=${root_dir}/ascent/src
ascent_build_dir=${root_dir}/build/ascent-${ascent_version}/
ascent_install_dir=${root_dir}/install/ascent-${ascent_version}/

if ${build_ascent}; then
if [ ! -d ${ascent_src_dir} ]; then
    echo "**** Cloning Ascent"
    git clone --recursive https://github.com/Alpine-DAV/ascent.git --branch task/2022_05_vtk_m_1.8_update
fi#!/bin/bash

##############################################################################
# Demonstrates how to manually build Ascent and its dependencies.
#
#
# Assumes: `cmake` is in your path
#           Selected compilers are in path or env vars
##############################################################################
set -eu -o pipefail

##############################################################################
# Build Options
##############################################################################

# shared options
enable_fortran="${enable_fortran:=OFF}"
enable_python="${enable_python:=OFF}"
enable_openmp="${enable_openmp:=OFF}"
enable_mpi="${enable_mpi:=OFF}"
enable_tests="${enable_tests:=ON}"

# tpls
build_hdf5="${build_hdf5:=true}"
build_conduit="${build_conduit:=true}"
build_vtkm="${build_vtkm:=true}"
build_raja="${build_raja:=true}"
build_umpire="${build_umpire:=true}"

# ascent options
build_ascent="${build_camp:=true}"

build_jobs="${build_jobs:=6}"

# HARDWIRE
# build_hdf5=false
# build_conduit=false
# build_vtkm=false
# build_raja=true
# build_umpire=false
# build_ascent=true

root_dir=$(pwd)

################
# HDF5
################
hdf5_version=1.12.2
hdf5_src_dir=${root_dir}/hdf5-${hdf5_version}
hdf5_build_dir=${root_dir}/build/hdf5-${hdf5_version}/
hdf5_install_dir=${root_dir}/install/hdf5-${hdf5_version}/
hdf5_tarball=hdf5-${hdf5_version}.tar.gz

if ${build_hdf5}; then
if [ ! -d ${hdf5_src_dir} ]; then
  echo "**** Downloading ${hdf5_tarball}"
  curl -L https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.gz -o ${hdf5_tarball}
  tar -xzf ${hdf5_tarball}
fi

echo "**** Configuring HDF5 ${hdf5_version}"
cmake -S ${hdf5_src_dir} -B ${hdf5_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${hdf5_install_dir}

echo "**** Building HDF5 ${hdf5_version}"
cmake --build ${hdf5_build_dir} -j${build_jobs}
echo "**** Installing HDF5 ${hdf5_version}"
cmake --install ${hdf5_build_dir}

fi # build_hdf5

################
# Conduit
################
conduit_version=v0.8.3
conduit_src_dir=${root_dir}/conduit-${conduit_version}/src
conduit_build_dir=${root_dir}/build/conduit-${conduit_version}/
conduit_install_dir=${root_dir}/install/conduit-${conduit_version}/
conduit_tarball=conduit-${conduit_version}-src-with-blt.tar.gz

if ${build_conduit}; then
if [ ! -d ${conduit_src_dir} ]; then
  echo "**** Downloading ${conduit_tarball}"
  curl -L https://github.com/LLNL/conduit/releases/download/${conduit_version}/${conduit_tarball} -o ${conduit_tarball}
  tar -xzf ${conduit_tarball}
fi

echo "**** Configuring Conduit ${conduit_version}"
cmake -S ${conduit_src_dir} -B ${conduit_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${conduit_install_dir} \
  -DENABLE_FORTRAN=${enable_fortran} \
  -DENABLE_MPI=${enable_mpi} \
  -DENABLE_PYTHON=${enable_python} \
  -DENABLE_TESTS=${enable_tests} \
  -DHDF5_DIR=${hdf5_install_dir}

echo "**** Building Conduit ${conduit_version}"
cmake --build ${conduit_build_dir} -j${build_jobs}
echo "**** Installing Conduit ${conduit_version}"
cmake --install ${conduit_build_dir}

fi # build_conduit

################
# VTK-m
################
vtkm_version=v1.8.0
vtkm_src_dir=${root_dir}/vtk-m-${vtkm_version}
vtkm_build_dir=${root_dir}/build/vtk-m-${vtkm_version}
vtkm_install_dir=${root_dir}/install/vtk-m-${vtkm_version}/
vtkm_tarball=vtk-m-${vtkm_version}.tar.gz

if ${build_vtkm}; then
if [ ! -d ${vtkm_src_dir} ]; then
  echo "**** Downloading ${vtkm_tarball}"
  curl -L https://gitlab.kitware.com/vtk/vtk-m/-/archive/${vtkm_version}/${vtkm_tarball} -o ${vtkm_tarball}
  tar -xzf ${vtkm_tarball}
fi

echo "**** Configuring VTK-m ${vtkm_version}"
cmake -S ${vtkm_src_dir} -B ${vtkm_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF\
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON\
  -DVTKm_NO_DEPRECATED_VIRTUAL=ON \
  -DVTKm_USE_64BIT_IDS=OFF \
  -DVTKm_USE_DOUBLE_PRECISION=ON \
  -DVTKm_USE_DEFAULT_TYPES_FOR_ASCENT=ON \
  -DVTKm_ENABLE_MPI=OFF \
  -DVTKm_ENABLE_RENDERING=ON \
  -DVTKm_ENABLE_TESTING=OFF \
  -DBUILD_TESTING=OFF \
  -DVTKm_ENABLE_BENCHMARKS=OFF\
  -DCMAKE_INSTALL_PREFIX=${vtkm_install_dir}

echo "**** Building VTK-m ${vtkm_version}"
cmake --build ${vtkm_build_dir} -j6
echo "**** Installing VTK-m ${vtkm_version}"
cmake --install ${vtkm_build_dir}

fi # build_vtkm


################
# RAJA
################
raja_version=v0.14.1
raja_src_dir=${root_dir}/RAJA-${raja_version}
raja_build_dir=${root_dir}/build/raja-${raja_version}
raja_install_dir=${root_dir}/install/raja-${raja_version}/
raja_tarball=RAJA-${raja_version}.tar.gz

if ${build_raja}; then
if [ ! -d ${raja_src_dir} ]; then
  echo "**** Downloading ${raja_tarball}"
  curl -L https://github.com/LLNL/RAJA/releases/download/${raja_version}/${raja_tarball} -o ${raja_tarball} 
  tar -xzf ${raja_tarball} 
fi

echo "**** Configuring RAJA ${raja_version}"
cmake -S ${raja_src_dir} -B ${raja_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF\
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON\
  -DENABLE_OPENMP=${enable_openmp} \
  -DENABLE_TESTS=${enable_tests} \
  -DCMAKE_INSTALL_PREFIX=${raja_install_dir}

echo "**** Building RAJA ${raja_version}"
cmake --build ${raja_build_dir} -j6
echo "**** Installing RAJA ${raja_version}"
cmake --install ${raja_build_dir}

fi # build_raja

################
# Umpire
################
umpire_version=6.0.0
umpire_src_dir=${root_dir}/Umpire-${umpire_version}
umpire_build_dir=${root_dir}/build/umpire-${umpire_version}
umpire_install_dir=${root_dir}/install/umpire-${umpire_version}/
umpire_tarball=umpire-${umpire_version}.tar.gz

if ${build_umpire}; then
if [ ! -d ${umpire_src_dir} ]; then
  echo "**** Downloading ${umpire_tarball}"
  curl -L https://github.com/LLNL/Umpire/releases/download/v${umpire_version}/${umpire_tarball} -o ${umpire_tarball}
  tar -xzf ${umpire_tarball}
fi

echo "**** Configuring Umpire ${umpire_version}"
cmake -S ${umpire_src_dir} -B ${umpire_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF\
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DENABLE_OPENMP=${enable_openmp} \
  -DENABLE_TESTS=${enable_tests} \
  -DCMAKE_INSTALL_PREFIX=${umpire_install_dir}

echo "**** Building Umpire ${umpire_version}"
cmake --build ${umpire_build_dir} -j6
echo "**** Installing Umpire ${umpire_version}"
cmake --install ${umpire_build_dir}

fi # build_umpire


################
# Ascent
################
ascent_version=develop
ascent_src_dir=${root_dir}/ascent/src
ascent_build_dir=${root_dir}/build/ascent-${ascent_version}/
ascent_install_dir=${root_dir}/install/ascent-${ascent_version}/

if ${build_ascent}; then
if [ ! -d ${ascent_src_dir} ]; then
    echo "**** Cloning Ascent"
    git clone --recursive https://github.com/Alpine-DAV/ascent.git --branch task/2022_05_vtk_m_1.8_update
fi

echo "**** Configuring Ascent"
cmake -S ${ascent_src_dir} -B ${ascent_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${ascent_install_dir} \
  -DENABLE_MPI=${enable_mpi} \
  -DENABLE_FORTRAN=${enable_fortran} \
  -DENABLE_TESTS=$enable_tests \
  -DENABLE_PYTHON=${enable_python} \
  -DBLT_CXX_STD=c++14 \
  -DCONDUIT_DIR=${conduit_install_dir} \
  -DVTKM_DIR=${vtkm_install_dir} \
  -DENABLE_VTKH=ON \
  -DRAJA_DIR=${raja_install_dir} \
  -DUMPIRE_DIR=${umpire_install_dir} \
  -DENABLE_APCOMP=ON \
  -DENABLE_DRAY=ON

echo "**** Building Ascent"
cmake --build ${ascent_build_dir} -j6
echo "**** Installing Ascent"
cmake --install ${ascent_build_dir}

fi # build_ascent



echo "**** Configuring Ascent"
cmake -S ${ascent_src_dir} -B ${ascent_build_dir} \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${ascent_install_dir} \
  -DENABLE_MPI=${enable_mpi} \
  -DENABLE_FORTRAN=${enable_fortran} \
  -DENABLE_TESTS=$enable_tests \
  -DENABLE_PYTHON=${enable_python} \
  -DBLT_CXX_STD=c++14 \
  -DCONDUIT_DIR=${conduit_install_dir} \
  -DVTKM_DIR=${vtkm_install_dir} \
  -DENABLE_VTKH=ON \
  -DRAJA_DIR=${raja_install_dir} \
  -DUMPIRE_DIR=${umpire_install_dir} \
  -DENABLE_APCOMP=ON \
  -DENABLE_DRAY=ON

echo "**** Building Ascent"
cmake --build ${ascent_build_dir} -j6
echo "**** Installing Ascent"
cmake --install ${ascent_build_dir}

fi # build_ascent

