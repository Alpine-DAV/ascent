#!/bin/bash
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
set -ev

export TAG_BASE=alpinedav/ascent-ci:ubuntu-22.04-devel-tpls

date

python ../build_and_tag.py ${TAG_BASE}

date

