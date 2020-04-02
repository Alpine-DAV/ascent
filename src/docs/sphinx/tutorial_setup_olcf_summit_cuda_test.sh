#!/bin/bash
#
# source helper script that loads the default gcc module, sets python paths, and ASCENT_DIR env var
#
source /gpfs/alpine/csc340/world-shared/software/ascent/current/summit/ascent_summit_setup_env_gcc_cuda.sh

#
# make your own dir to hold the tutorial examples
#
mkdir ascent_tutorial
cd ascent_tutorial

#
# copy the examples from the public install
#
cp -r /gpfs/alpine/world-shared/csc340/software/ascent/current/summit/cuda/gnu/ascent-install/examples/ascent/tutorial/* .

#
# build cpp examples and run the first one
#
cd ascent_intro/cpp
make
./ascent_first_light_example


