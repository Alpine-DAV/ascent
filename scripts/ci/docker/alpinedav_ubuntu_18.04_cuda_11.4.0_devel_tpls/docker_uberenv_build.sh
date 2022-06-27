#!/bin/bash
set -ev

# variants
# TODO:
# (+genten) genten currently disabled, wait for genten master to gain cokurt
# (^vtk-m@1.8.0-rc1) use vtk-m 1.8 rc branch  (currently 1.7.1)
export SPACK_SPEC="%gcc+mpi+cuda+vtkh+dray+mfem+occa~genten~python~openmp~shared cuda_arch=70"
# constraints
export SPACK_SPEC="${SPACK_SPEC} ^raja@0.14.0+cuda cuda_arch=70  ^dray~openmp+cuda cuda_arch=70 ^hdf5~mpi ^mfem~cuda~petsc~sundials~slepc ^hypre~cuda ^vtk-h+cuda cuda_arch=70 ^vtk-m+cuda~kokkos@1.7.1 cuda_arch=70"
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

# back to where we started
cd ../