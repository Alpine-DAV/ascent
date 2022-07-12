#!/bin/bash
################
set -ev

#
# base containers
#

echo "[BUILDING BASE CONTAINERS]"

#
# ubuntu 18
#
echo "[ubuntu 18.04 base]"
cd alpinedav_ubuntu_18.04_devel
./build.sh 
cd ..
# ubuntu 20.04
echo "[ubuntu 20.04 base]"
cd alpinedav_ubuntu_20.04_devel
./build.sh 
cd ..
# ubuntu 22.04
echo "[ubuntu 21.04 base]"
cd alpinedav_ubuntu_22.04_devel
./build.sh 
cd ..
#
# ubuntu 18.04 cuda 11.4.0
#
echo "[ubuntu 18.04 cuda 11.4.0 base]"
cd alpinedav_ubuntu_18.04_cuda_11.4.0_devel
./build.sh 
cd ..
#
# ubuntu 20.04 rocm 5.1.3
#
echo "[ubuntu 20.04 rocm 5.1.3 base]"
cd alpinedav_ubuntu_20.04_rocm_5.1.3_devel
./build.sh 
cd ..

echo "[BASE CONTAINERS BUILDS COMPLETE]"

