#!/bin/bash
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

# add spack's mpiexec to path
#export PATH=$PATH:`ls -d uberenv_libs/spack/opt/spack/*/*/mpich-*/bin/`

# run noise
build-debug/examples/synthetic/noise/noise_ser

# run kripke
mpiexec -n 2 build-debug/examples/proxies/kripke/kripke_par --procs 2,1,1  --zones 32,32,32 --niter 3 --dir 1:12 --grp 1:1 --legendre 4 --quad 20:20 --dir 50:1

# run clover
cp build-debug/examples/proxies/cloverleaf3d-ref/clover.in .
mpiexec -n 2 build-debug/examples/proxies/cloverleaf3d-ref/cloverleaf3d_par

# run lulesh
mpiexec -n 8 build-debug/examples/proxies/lulesh2.0.3/lulesh_par -p -i 5


if [ -d "build-debug/examples/proxies/laghos" ]; then
  ./build-debug/examples/proxies/laghos/laghos_ser -rs 1 -tf 0.2 -m build-debug/examples/proxies/laghos/data/square01_quad.mesh --visit
fi
