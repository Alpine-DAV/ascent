#!/bin/bash
#set -ev

# variants
# TODO:
# (+genten) genten currently disabled, wait for genten master to gain cokurt
export SPACK_SPEC="%clang+mpi+vtkh+dray+mfem+occa+rocm~genten~python~openmp~fortran"
# constraints
export SPACK_SPEC="${SPACK_SPEC} ^hdf5~mpi ^mfem~rocm ^hypre~rocm ^raja@2022.03.0 ^conduit~fortran"
# config
export SPACK_CONFIG="scripts/uberenv_configs/spack_configs/configs/alpinedav/ubuntu_20.04_rocm_5.1.3_devel/"

cd ascent && python scripts/uberenv/uberenv.py \
     -k \
     --spec="${SPACK_SPEC}" \
     --spack-config-dir="${SPACK_CONFIG}" \
     --prefix="/uberenv_libs"

# cleanup the spack build stuff to free up space
/uberenv_libs/spack/bin/spack clean --all

# change perms
chmod -R a+rX /uberenv_libs

# back to where we started
cd ../