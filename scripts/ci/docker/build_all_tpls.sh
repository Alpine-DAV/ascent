#!/bin/bash
################
set -ev

#
# TPL BUILDS
#

echo "[BUILDING TPL CONTAINERS]"

# ubuntu 18.04 tpls
echo "[ubuntu 18.04 tpls]"
cd alpinedav_ubuntu_18.04_devel_tpls
./build.sh
cd ..
# ubuntu 20.04
echo "[ubuntu 20.04 tpls]"
cd alpinedav_ubuntu_20.04_devel_tpls
./build.sh 
cd ..
# ubuntu 22.04 (on hold until we get spack issue sorted)
#echo "[ubuntu 22.04 tpls]"
#cd alpinedav_ubuntu_22.04_devel_tpls
#./build.sh 
#cd ..
# ubuntu 18.04 cuda 11.4.0 tpls
echo "[ubuntu 18.04 cuda 11.4.0 tpls]"
cd alpinedav_ubuntu_18.04_cuda_11.4.0_devel_tpls
./build.sh
cd ..

# ubuntu 20.04 rocm 5.1.3 tpls
echo "[ubuntu 20.04 rocm 5.1.3 tpls]"
cd alpinedav_ubuntu_20.04_rocm_5.1.3_devel_tpls
./build.sh
cd ..

echo "[TPL CONTAINERS BUILDS COMPLETE]"
