#!/bin/bash
set -ev

cmake_install_dir=/cmake-3.23.2-linux-x86_64
if [ ! -d ${cmake_install_dir} ]; then
  echo "**** Downloading ${hdf5_tarball}"
  # setup cmake in container
  curl -L https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-linux-x86_64.tar.gz -o cmake-3.23.2-linux-x86_64.tar.gz
  tar -xzf cmake-3.23.2-linux-x86_64.tar.gz
fi

export PATH=$PATH:/${cmake_install_dir}/bin/

# build cuda tpls with helper script
chmod +x ascent/scripts/build_ascent/build_ascent_cuda.sh
env build_ascent=false ascent/scripts/build_ascent/build_ascent_cuda.sh

