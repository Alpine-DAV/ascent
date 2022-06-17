#!/bin/bash
#set -ev

# variants
# TODO:
# (+genten) genten currently disabled, wait for genten master to gain cokurt
# (+dray) rocm support isn't in dray pkg, we need to move to intl ver and +raja opt
# (^vtk-m@1.8.0-rc1) use vtk-m 1.8 rc branch  (currently 1.7.1)
# (^vtk-h+rocm) rocm support isn't in vtk-h pkg yet
export SPACK_SPEC="%gcc+mpi+cuda+vtkh~dray+mfem+occa~genten~python~openmp~shared"
# constraints
export SPACK_SPEC="${SPACK_SPEC} ^hdf5~mpi ^kokkos+wrapper ^mfem~rocm ^hypre~rocm ^vtk-h ^vtk-m+rocm@1.7.1"
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