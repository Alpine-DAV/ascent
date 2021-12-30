#!/bin/bash
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


# Start the ubuntu-paraview-ascent docker and run
# build_and_run_sim_inside_docker.sh

# ascent dir inside docker. See README-docker.md for the command that builds the
# container
date
ascentDir=/root/projects/ascent

# keep_going: optional count that says how many time we keep going when we should stop
keep_going=$1
if [[ -z $keep_going ]]; then
    keep_going=0
fi

build_option=$2
if [[ -z $build_option ]]; then
    build_option="-j40"
fi

build_dependency=$3
if [[ -z $build_dependency ]]; then
    build_dependency=""
fi

docker container start ubuntu-paraview-ascent
docker exec ubuntu-paraview-ascent ${ascentDir}/src/examples/paraview-vis/tests/build_and_run_sim_inside_docker.sh $keep_going $build_option $build_dependency
date
