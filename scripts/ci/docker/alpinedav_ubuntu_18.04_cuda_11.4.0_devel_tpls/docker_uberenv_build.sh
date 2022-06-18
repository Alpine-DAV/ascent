#!/bin/bash
#set -ev

# variants
# TODO:
# (+genten) genten currently disabled, wait for genten master to gain cokurt
# (^vtk-m@1.8.0-rc1) use vtk-m 1.8 rc branch  (currently 1.7.1)
# (+dray ^dray~openmp+cuda) issue with umpire linking (solve in amlaga tests?)
export SPACK_SPEC="%gcc+mpi+cuda+vtkh~dray+mfem+occa~genten~python~openmp~shared"
# constraints
export SPACK_SPEC="${SPACK_SPEC} ^hdf5~mpi ^mfem~cuda~petsc~sundials~slep ^hypre~cuda ^vtk-h+cuda ^vtk-m+cuda~kokkos@1.7.1"
# config
export SPACK_SPEC="${SPACK_SPEC} ^hdf5~mpi ^dray~openmp+cuda c ^hypre~cuda ^vtk-h+cuda ^vtk-m+cuda~kokkos@1.7.1"
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