#!/bin/bash

# Runs inside docker container.
# Sets up spack and run
# build_and_run_sim.sh

# ascent dir inside docker
# working directory is /root
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

# Patch spack to fix eventual issues
cd spack || exit;git reset --hard;cd ..
if [[ -f ${ascentDir}/spack.patch ]]; then
    cd spack || exit;patch -p1 < ${ascentDir}/src/examples/paraview-vis/tests/spack.patch;cd ..
    . spack/share/spack/setup-env.sh
fi
# the env variable is needed because we run this as root
FORCE_UNSAFE_CONFIGURE=1 ${ascentDir}/src/examples/paraview-vis/tests/build_and_run_sim.sh /root/spack /root/tests $keep_going $build_option $build_dependency
