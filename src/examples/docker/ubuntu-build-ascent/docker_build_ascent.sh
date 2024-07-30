#!/bin/bash
set -ev

env enable_python=ON \
    enable_mpi=ON \
    enable_openmp=ON \
    build_pyvenv=true \
    /ascent/scripts/build_ascent/build_ascent.sh

# ascent is installed at /install/ascent-checkout
ln -s /install/ascent-checkout/ /ascent/install

# install extra packages
/install/python-venv/bin/python3 -m pip install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    bash_kernel \
    jupyter \
    jupyterlab \
    ipywidgets \
    ipympl \
    matplotlib \
    pyyaml \
    cinemasci \
    scipy \
    scikit-learn \
    h5py \
    llnl-hatchet


# install ascent bridge kernel into jupyter
cd /install/ascent-checkout/share/ascent/ascent_jupyter_bridge/ && \
    /install/python-venv/bin/python3 -m pip install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt

cd /install/ascent-checkout/share/ascent/ascent_jupyter_bridge/ && \
    /install/python-venv/bin/python3 -m ipykernel install


echo "export PATH=/install/python-venv/bin/:$PATH" >> /ascent_docker_setup_env.sh
echo "export PYTHONPATH=/ascent/install/python-modules/" >> /ascent_docker_setup_env.sh
