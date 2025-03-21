#!/bin/bash
################
set -ev

#
# base containers
#

echo "[BUILDING BASE CONTAINERS]"

# ubuntu 22.04
echo "[ubuntu 22.04 base]"
cd alpinedav_ubuntu_22.04_devel
./build.sh
cd ..

#
# ubuntu 22.04 cuda 11
#
echo "[ubuntu 22.04 cuda 11.8.0 base]"
cd alpinedav_ubuntu_22.04_cuda_11.8.0_devel
./build.sh
cd ..

#
# ubuntu 22.04 cuda 12
#
echo "[ubuntu 22.04 cuda 12.8.1 base]"
cd alpinedav_ubuntu_22.04_cuda_12.8.1_devel
./build.sh
cd ..

#
# ubuntu 22.04 rocm 6.3.0
#
echo "[ubuntu 22.04 rocm 6.3.0 base]"
cd alpinedav_ubuntu_22.04_rocm_6.3.0_devel
./build.sh
cd ..

echo "[BASE CONTAINERS BUILDS COMPLETED]"

