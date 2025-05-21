#!/bin/bash
set -ev

# variants
# TODO:
# (+genten) genten currently disabled, wait for genten master to gain cokurt
export SPACK_SPEC="%gcc+mpi+cuda+vtkh+dray+mfem+occa~python~openmp~shared cuda_arch=70"
# constraints
# note: silo static build with spack will fail unless you disable python
export SPACK_SPEC="${SPACK_SPEC} ^raja+cuda~openmp cuda_arch=70 ^umpire+cuda~openmp cuda_arch=70 ^camp+cuda~openmp cuda_arch=70 ^hdf5~mpi ^mfem~cuda~petsc~sundials~slepc ^hypre~cuda ^vtk-m+cuda~kokkos cuda_arch=70 ^silo~python~mpi"
# config
export SPACK_ENV_YAML="scripts/uberenv_configs/spack_configs/envs/alpinedav-ubuntu_22.04_cuda_11.8.0_devel/spack.yaml"

cd ascent && python3 scripts/uberenv/uberenv.py \
     -k \
     -j 6 \
     --spec="${SPACK_SPEC}" \
     --spack-env-file="${SPACK_ENV_YAML}" \
     --prefix="/uberenv_libs"

# cleanup the spack build stuff to free up space
/uberenv_libs/spack/bin/spack clean --all

# change perms
chmod -R a+rX /uberenv_libs

# back to where we started
cd ../