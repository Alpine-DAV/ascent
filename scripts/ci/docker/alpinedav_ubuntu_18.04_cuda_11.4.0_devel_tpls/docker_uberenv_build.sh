#!/bin/bash
set -ev

# variants
# TODO:
# (+genten) genten currently disabled, wait for genten master to gain cokurt
# (^vtk-m@1.8.0-rc1) use vtk-m 1.8 rc branch  (currently 1.7.1)
export SPACK_SPEC="%gcc+mpi+cuda+vtkh+dray+mfem+occ~genten~python~openmp~shared"
# constraints
export SPACK_SPEC="${SPACK_SPEC} ^hdf5~mpi ^dray~openmp+cuda ^kokkos+wrapper ^mfem~cuda cuda_arch=none ^hypre~cuda cuda_arch=none  ^vtk-h+cuda ^vtk-m+cuda@1.7.1 ^cmake~openssl~ncurses"
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
