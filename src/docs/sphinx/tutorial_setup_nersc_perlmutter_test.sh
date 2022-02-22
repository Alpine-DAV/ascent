#!/bin/bash
#
# source helper script that loads modules, sets python paths, and ASCENT_DIR env var
#
source /project/projectdirs/alpine/software/ascent/current/cori/ascent_permutter_setup_env_gcc_cuda.sh

#
# make your own dir to hold the tutorial examples
#
mkdir ascent_tutorial
cd ascent_tutorial

#
# copy the examples from the public install
#
cp -r /global/cfs/cdirs/alpine/software/ascent/current/perlmutter/cuda/gnu/ascent-install/examples/ascent/tutorial/* .

#
# build cpp examples and run the first one
#
cd ascent_intro/cpp
make
./ascent_first_light_example

