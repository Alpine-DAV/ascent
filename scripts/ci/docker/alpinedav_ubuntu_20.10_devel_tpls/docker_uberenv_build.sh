#!/bin/bash
set -ev

# variants
# NOTE: fides needs a new release for vtk-m 1.7.0
export SPACK_SPEC="%gcc+mpi+python+babelflow+dray+mfem+occa"
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
