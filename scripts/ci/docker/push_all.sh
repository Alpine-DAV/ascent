#!/bin/bash
################
#
# ubuntu 18
#
docker push alpinedav/ascent-ci:ubuntu-18-devel
docker push alpinedav/ascent-ci:ubuntu-18-devel-tpls

#
# cuda 9.2 
#
docker push alpinedav/ascent-ci:ubuntu-16-cuda-9.2-devel
docker push alpinedav/ascent-ci:ubuntu-16-cuda-9.2-devel-tpls
