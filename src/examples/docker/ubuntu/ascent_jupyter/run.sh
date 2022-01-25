#!/bin/bash
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

# exec docker run to create a container from our image
echo "docker run -p 8000:8000 -p 8888:8888 -p 9000:9000 -p 10000:10000 -t -i alpinedav/ascent-jupyter:latest"
docker run -p 8000:8000 -p 8888:8888 -p 9000:9000 -p 10000:10000 -t -i alpinedav/ascent-jupyter:latest
