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
echo "[ubuntu 18 base]"
cd alpinedav_ubuntu_18_devel
./build.sh 
cd ..
# ubuntu 20.04
echo "[ubuntu 20.04 base]"
cd alpinedav_ubuntu_20.04_devel
./build.sh 
cd ..
# ubuntu 21.04
echo "[ubuntu 21.04 base]"
cd alpinedav_ubuntu_21.04_devel
./build.sh 
cd ..
# ubuntu 21.10
echo "[ubuntu 21.10 base]"
cd alpinedav_ubuntu_21.10_devel
./build.sh 
cd ..
#
# ubuntu 18 cuda 10.1
#
echo "[ubuntu 18 cuda 10.1 base]"
cd alpinedav_ubuntu_18_cuda_10.1_devel
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
# ubuntu 20.04 rocm 4.5.0
#
echo "[ubuntu 20.04 rocm 4.5.0 base]"
cd alpinedav_ubuntu_20.04_rocm_4.5.0_devel
./build.sh 
cd ..

echo "[BASE CONTAINERS BUILDS COMPLETE]"

