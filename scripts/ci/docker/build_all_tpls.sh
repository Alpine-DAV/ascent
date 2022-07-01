#!/bin/bash
################
set -ev

#
# TPL BUILDS
#

echo "[BUILDING TPL CONTAINERS]"

# ubuntu 18 tpls
echo "[ubuntu 18 tpls]"
cd alpinedav_ubuntu_18_devel_tpls
./build.sh
cd ..
# ubuntu 20.04
echo "[ubuntu 20.04 tpls]"
cd alpinedav_ubuntu_20.04_devel_tpls
./build.sh 
cd ..
# ubuntu 21.04 tpls
echo "[ubuntu 21.04 tpls]"
cd alpinedav_ubuntu_21.04_devel_tpls
./build.sh
cd ..
# ubuntu 21.10 tpls
echo "[ubuntu 21.10 tpls]"
cd alpinedav_ubuntu_21.10_devel_tpls
./build.sh
cd ..
# ubuntu 18 cuda 10.1 tpls
echo "[ubuntu 18 cuda 10.1 tpls]"
cd alpinedav_ubuntu_18_cuda_10.1_devel_tpls
./build.sh
cd ..
# ubuntu 18.04 cuda 11.4.0 tpls
echo "[ubuntu 18.04 cuda 11.4.0 tpls]"
cd alpinedav_ubuntu_18.04_cuda_11.4.0_devel_tpls
./build.sh
cd ..

echo "[TPL CONTAINERS BUILDS COMPLETE]"
