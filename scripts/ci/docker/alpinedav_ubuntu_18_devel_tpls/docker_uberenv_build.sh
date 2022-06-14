#!/bin/bash
#set -ev

# variants
# TODO:
# (+python) python currently disabled, spack checksum fetch issues with multiple versions (3.9 and 3.8)
# (+genten) genten currently disabled, wait for genten master to gain cokurt
# (^vtk-m@1.8.0-rc1) use vtk-m 1.8 rc branch  (currently 1.7.1)
export SPACK_SPEC="%gcc+mpi~python+dray+mfem+occa+fides+adios2+babelflow~genten ^vtk-m@1.7.1"
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
