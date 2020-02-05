#!/bin/bash

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

docker container start ubuntu-paraview-ascent
docker exec ubuntu-paraview-ascent ${ascentDir}/src/examples/paraview-vis/tests/build_and_run_sim_inside_docker.sh $keep_going
date
