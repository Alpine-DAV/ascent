#!/bin/bash
################
#
# ubuntu 18
#
cd alpinedav_ubuntu_18_devel
./build.sh 
cd ..
cd alpinedav_ubuntu_18_devel_tpls
./build.sh
#
# cuda 9.2 
#
cd alpinedav_ubuntu_16_cuda_10.2_devel
./build.sh 
cd ..
cd alpinedav_ubuntu_16_cuda_10.2_devel_tpls
./build.sh

