#!/bin/bash
set -e
#
# run at root of ascent repo
#
date
# run spack install, this will install ascent@develop
export ASCENT_VERSION=0.8.0_genten
export BASE_DIR=$PSCRATCH/ASCENT_INSTALL 
#/project/projectdirs/alpine/software/ascent/
export DEST_DIR=$BASE_DIR/${ASCENT_VERSION}/
mkdir -p $DEST_DIR
# ^vtk-h~blt_find_mpi
python3 scripts/uberenv/uberenv.py --spec="%gcc~blt_find_mpi+genten ^conduit~blt_find_mpi ^vtk-h~blt_find_mpi ^kokkos+wrapper  ^hdf5~mpi ^cmake~openssl~ncurses" \
       --pull \
       --install \
       --spack-config-dir="scripts/uberenv_configs/spack_configs/configs/nersc/perlmutter_gcc_10.3.0_cuda_11.4/" \
       --prefix=${DEST_DIR}

# gen symlinks to important deps
python3 scripts/spack_install/gen_extra_install_symlinks.py ${DEST_DIR} cmake conduit
# gen env helper script
rm -f public_env.sh
python3 scripts/spack_install/gen_public_install_env_script.py ${DEST_DIR} PrgEnv-gnu cpe-cuda/21.12  cudatoolkit/21.9_11.4 
chmod a+x public_env.sh
cp public_env.sh $BASE_DIR/${ASCENT_VERSION}/ascent_permutter_setup_env_gcc_cuda.sh
# change perms to world readable
chmod a+rX -R /project/projectdirs/alpine/software/ascent/${ASCENT_VERSION}/
date

