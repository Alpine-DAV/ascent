#!/bin/bash
set -ev

# variants
export SPACK_SPEC="%gcc+mpi+python+babelflow+fides+adios2+dray+mfem+occa"
# constraints
export SPACK_SPEC="${SPACK_SPEC} ^conduit@develop ^vtk-h@develop ^dray@develop~test~utils"
# config
export SPACK_CONFIG="scripts/uberenv_configs/spack_configs/configs/alpinedav/ubuntu_18_devel/"

cd /home/user/ascent && python scripts/uberenv/uberenv.py \
     -k \
     --spec="${SPACK_SPEC}" \
     --spack-config-dir="${SPACK_CONFIG}" \
     --prefix="/uberenv_libs"

# cleanup the spack build stuff to free up space
/home/user/uberenv_libs/spack/bin/spack clean --all

# change perms
chmod -R a+rX /home/user/uberenv_libs
