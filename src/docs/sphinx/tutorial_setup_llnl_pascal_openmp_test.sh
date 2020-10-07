#!/bin/bash
#
# source helper script that loads the default gcc module, sets python paths, and ASCENT_DIR env var
#
source /usr/gapps/conduit/software/ascent/current/toss_3_x86_64_ib/ascent_toss_3_x86_64_ib_setup_env_gcc_openmp.sh

#
# make your own dir to hold the tutorial examples
#
mkdir ascent_tutorial
cd ascent_tutorial

#
# copy the examples from the public install
#
cp -r /usr/gapps/conduit/software/ascent/current/toss_3_x86_64_ib/openmp/gnu/ascent-install/examples/ascent/tutorial/* .

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

