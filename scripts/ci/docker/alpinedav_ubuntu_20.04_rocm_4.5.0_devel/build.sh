#!/bin/bash
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
set -ev

export TAG_NAME=alpinedav/ascent-ci:ubuntu-20.04-rocm-4.5.0-devel

# exec docker build to create image
echo "docker build -t ${TAG_NAME} ."
docker build -t ${TAG_NAME} .
