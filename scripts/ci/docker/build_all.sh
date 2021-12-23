#!/bin/bash
################
set -ev
#
# ubuntu 18
#
cd alpinedav_ubuntu_18_devel
./build.sh 
cd ..
# + tpls
cd alpinedav_ubuntu_18_devel_tpls
./build.sh
cd ..

# ubuntu 20.10
cd alpinedav_ubuntu_20.10_devel
./build.sh 
cd ..
# + tpls
cd alpinedav_ubuntu_20.10_devel_tpls
./build.sh
cd ..

# ubuntu 21.04
cd alpinedav_ubuntu_21.04_devel
./build.sh 
cd ..
# + tpls
cd alpinedav_ubuntu_21.04_devel_tpls
./build.sh
cd ..

#
# ubuntu 18 cuda 10.1
#
cd alpinedav_ubuntu_18_cuda_10.1_devel
./build.sh 
cd ..
# + tpls
cd alpinedav_ubuntu_18_cuda_10.1_devel_tpls
./build.sh
cd ..

#
# ubuntu 18.04 cuda 11.4.0
#
cd alpinedav_ubuntu_18.04_cuda_11.4.0_devel
./build.sh 
cd ..
# + tpls
cd alpinedav_ubuntu_18.04_cuda_11.4.0_devel_tpls
./build.sh
cd ..

