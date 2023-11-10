#!/bin/bash
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
set -ev

export TAG_ARCH=`uname -m`
export TAG_BASE=alpinedav/ascent-devel:ubuntu-22.04-${TAG_ARCH}-tpls

echo "TAG_BASE: ${TAG_BASE}"

date

python3 ../build_and_tag.py ${TAG_BASE}

date

