#!/bin/bash
set -ev

# variants
# NOTES:
#  genten currently disabled, genten needs a spack package
export SPACK_SPEC="%gcc+mpi+python+dray+caliper+occa+mfem+fides+adios2+babelflow ^occa~cuda"
# constraints
export SPACK_SPEC="${SPACK_SPEC}"
# config
export SPACK_ENV_YAML="scripts/uberenv_configs/spack_configs/envs/alpinedav-ubuntu_18.04_devel/spack.yaml"

cd ascent && python3 scripts/uberenv/uberenv.py \
     -k \
     --spec="${SPACK_SPEC}" \
     --spack-env-file="${SPACK_ENV_YAML}" \
     --prefix="/uberenv_libs"

# cleanup the spack build stuff to free up space
/uberenv_libs/spack/bin/spack clean --all

# change perms
chmod -R a+rX /uberenv_libs

# back to where we started
cd ../
