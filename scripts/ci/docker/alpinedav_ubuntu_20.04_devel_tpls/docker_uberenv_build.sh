#!/bin/bash
set -ev

# variants
export SPACK_SPEC="%gcc+mpi+python+dray+mfem+occa+fides+adios2+babelflow+genten"
# constraints
export SPACK_SPEC="${SPACK_SPEC}"
# config
export SPACK_CONFIG="scripts/uberenv_configs/spack_configs/configs/alpinedav/ubuntu_18_devel/"

cd ascent && python scripts/uberenv/uberenv.py \
     -k \
     --spec="${SPACK_SPEC}" \
     --spack-config-dir="${SPACK_CONFIG}" \
     --prefix="/uberenv_libs"

# cleanup the spack build stuff to free up space
/uberenv_libs/spack/bin/spack clean --all

# change perms
chmod -R a+rX /uberenv_libs
