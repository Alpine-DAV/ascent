#!/bin/bash
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

export REPO_NAME="ascent"
export TAG_ARCH=`uname -m`
export TAG_BASE=alpinedav/ascent:ascent-ubuntu-22.04-${TAG_ARCH}

date

python3 ../../../../scripts/docker_build_and_tag.py ${REPO_NAME} ${TAG_ARCH} ${TAG_BASE}

date

