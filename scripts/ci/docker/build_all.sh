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
cd ..

#
# cuda 10.1
#
cd alpinedav_ubuntu_16_cuda_10.1_devel
./build.sh 
cd ..
cd alpinedav_ubuntu_16_cuda_10.1_devel_tpls
./build.sh

