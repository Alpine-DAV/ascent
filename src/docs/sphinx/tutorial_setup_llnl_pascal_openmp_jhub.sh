#!/bin/bash
#
# source helper script that loads modules, sets python paths, and ASCENT_DIR env var
#
source /usr/gapps/conduit/software/ascent/current/toss_3_x86_64_ib/ascent_toss_3_x86_64_ib_setup_env_gcc_openmp.sh

#
# Register Ascent's Python with Jupyter Hub
#
python -m ipykernel install --user --name ascent_kernel --display-name "Ascent Kernel"
