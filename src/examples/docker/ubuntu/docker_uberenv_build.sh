#!/bin/bash
set -ev

# variants
export SPACK_SPEC="%gcc+mpi+python+babelflow+fides+adios2+dray+mfem+occa"
# constraints
export SPACK_SPEC="${SPACK_SPEC} ^conduit@develop ^vtk-h@develop ^dray@develop~test~utils"
# config
export SPACK_CONFIG="scripts/uberenv_configs/spack_configs/configs/alpinedav/ubuntu_18_devel/"

cd /ascent && python scripts/uberenv/uberenv.py \
     -k \
     --spec="${SPACK_SPEC}" \
     --spack-config-dir="${SPACK_CONFIG}" \
     --prefix="/uberenv_libs"

# cleanup the spack build stuff to free up space
/uberenv_libs/spack/bin/spack clean --all

# create some helper scripts
# clone script
echo "git clone --recursive https://github.com/Alpine-DAV/ascent.git" > clone.sh
chmod +x clone.sh

#  gen env script that points to spack installs of tpls
cd /uberenv_libs/ && python /ascent/scripts/gen_spack_env_script.py cmake mpi python
cp /uberenv_libs/s_env.sh /ascent_docker_setup_env.sh
echo "export PYTHONPATH=/ascent/install-debug/python-modules/" >> /ascent_docker_setup_env.sh
