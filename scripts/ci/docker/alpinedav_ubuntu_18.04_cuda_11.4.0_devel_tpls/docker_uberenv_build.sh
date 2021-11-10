#!/bin/bash
set -ev

# variants
export SPACK_SPEC="%gcc+mpi+cuda+vtkh+dray+mfem+occa~python~openmp~shared"
# constraints
export SPACK_SPEC="${SPACK_SPEC} ^hdf5~mpi ^dray@develop~openmp+cuda ^mfem~cuda cuda_arch=none ^hypre~cuda cuda_arch=none ^conduit@develop ^vtk-h@develop+cuda ^vtk-m+cuda ^cmake~openssl~ncurses"
# config
export SPACK_CONFIG="scripts/uberenv_configs/spack_configs/configs/alpinedav/ubuntu_18.04_cuda_11.4.0_devel/"

cd ascent && python scripts/uberenv/uberenv.py \
     -k \
     --spec="${SPACK_SPEC}" \
     --spack-config-dir="${SPACK_CONFIG}" \
     --prefix="/uberenv_libs"

# cleanup the spack build stuff to free up space
/uberenv_libs/spack/bin/spack clean --all

# change perms
chmod -R a+rX /uberenv_libs
