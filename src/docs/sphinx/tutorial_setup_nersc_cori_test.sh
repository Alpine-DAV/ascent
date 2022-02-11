#!/bin/bash
#
# source helper script that loads modules, sets python paths, and ASCENT_DIR env var
#
source /project/projectdirs/alpine/software/ascent/current/cori/ascent_cori_setup_env_gcc.sh

#
# make your own dir to hold the tutorial examples
#
mkdir ascent_tutorial
cd ascent_tutorial

#
# copy the examples from the public install
#
cp -r /project/projectdirs/alpine/software/ascent/current/cori/gnu/ascent-install/examples/ascent/tutorial/* .

#
# build cpp examples and run the first one
#
cd ascent_intro/cpp
make
env OMP_NUM_THREADS=1 ./ascent_first_light_example

#
# run a python example
#
cd ..
cd python
env OMP_NUM_THREADS=1 python ascent_first_light_example.py
