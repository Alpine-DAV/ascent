#!/bin/bash
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
set -ev

export TAG_ARCH=`uname -m`
export TAG_NAME=alpinedav/ascent-devel:ubuntu-22.04-cuda-12.8.1-${TAG_ARCH}

# exec docker build to create image
echo "docker build -t ${TAG_NAME} ."
docker build -t ${TAG_NAME} .
