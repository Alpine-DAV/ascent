#!/bin/bash
set -e
#
# run at root of ascent repo
#
date
# run spack install, this will install ascent@develop
export ASCENT_VERSION=0.5.2-pre
export DEST_DIR=/project/projectdirs/alpine/software/ascent/${ASCENT_VERSION}/cori/gnu
mkdir -p $DEST_DIR
python scripts/uberenv/uberenv.py --spec="%gcc" \
       --pull \
       --install \
       --spack-config-dir="scripts/uberenv/spack_configs/nersc/cori/" \
       --prefix=${DEST_DIR}

# gen env helper script
rm public_env.sh
python scripts/spack_install/gen_public_install_env_script.py ${DEST_DIR} gcc/8.2.0
chmod a+x public_env.sh
cp public_env.sh /project/projectdirs/alpine/software/ascent/${ASCENT_VERSION}/cori/ascent_cori_setup_env_gcc.sh
# change perms to world readable
chmod a+rX -R /project/projectdirs/alpine/software/ascent/${ASCENT_VERSION}/
date
