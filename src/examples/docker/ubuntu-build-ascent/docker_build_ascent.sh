#!/bin/bash
set -ev

env enable_python=ON \
    enable_mpi=ON \
    enable_FORTRAN=ON \
    enable_openmp=ON \
    build_pyvenv=true \
    /ascent/scripts/build_ascent/build_ascent.sh

# install extra python modules
/install/python-venv/bin/python3 -m pip install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    bash_kernel \
    jupyter \
    jupyterlab \
    ipykernel \
    pylint \
    ipywidgets \
    ipympl \
    matplotlib \
    pyyaml \
    cinemasci \
    scipy \
    scikit-learn \
    h5py \
    llnl-hatchet

cd /install/ascent-checkout/share/ascent/ascent_jupyter_bridge/ && \
    /install/python-venv/bin/python3 -m ipykernel install

vtkm_lib_path=`ls -d /install/vtk-m-*/lib`
echo "export PATH=/install/python-venv/bin/:$PATH" >> /ascent_docker_setup_env.sh
echo "export LD_LIBRARY_PATH=${vtkm_lib_path}" >> /ascent_docker_setup_env.sh

