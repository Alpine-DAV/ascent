#!/bin/bash
set -ev

export CMAKE_ARCH=`uname -m`

if [[ ${CMAKE_ARCH} == "arm64" ]]; then
  export CMAKE_ARCH="aarch64"
fi

cmake_install_dir=/cmake-3.23.2-linux-${CMAKE_ARCH}
if [ ! -d ${cmake_install_dir} ]; then
  # setup cmake in container
  curl -L https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-linux-${CMAKE_ARCH}.tar.gz -o cmake-3.23.2-linux-${CMAKE_ARCH}.tar.gz
  tar -xzf cmake-3.23.2-linux-${CMAKE_ARCH}.tar.gz
fi

export PATH=$PATH:/${cmake_install_dir}/bin/

export CFLAGS="-fsanitize=address"
export CXXFLAGS="-fsanitize=address"

# build tpls with helper script /w clang and 
chmod +x ascent/scripts/build_ascent/build_ascent.sh
# bi
env CXX=clang++ CC=clang enable_tests=OFF build_ascent=false ascent/scripts/build_ascent/build_ascent.sh

