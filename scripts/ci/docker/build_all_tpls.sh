#!/bin/bash
################
set -ev

#
# TPL BUILDS
#

echo "[BUILDING TPL CONTAINERS]"


# ubuntu 22.04
echo "[ubuntu 22.04 tpls]"
cd alpinedav_ubuntu_22.04_devel_tpls
./build.sh
cd ..

# ubuntu 22.04 cuda 11 tpls
echo "[ubuntu 22.04 cuda 11.8.0 tpls]"
cd alpinedav_ubuntu_22.04_cuda_11.8.0_devel_tpls
./build.sh
cd ..

# ubuntu 22.04 cuda 12 tpls
echo "[ubuntu 22.04 cuda 12.8.1 tpls]"
cd alpinedav_ubuntu_22.04_cuda_12.8.1_devel_tpls
./build.sh
cd ..

# ubuntu 22.04 rocm 6.3.0 tpls
echo "[ubuntu 22.04 rocm 6.3.0 tpls]"
cd alpinedav_ubuntu_22.04_rocm_6.3.0_devel_tpls
./build.sh
cd ..

echo "[TPL CONTAINERS BUILDS COMPLETED]"
