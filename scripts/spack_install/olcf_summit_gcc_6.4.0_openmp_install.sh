#!/bin/bash
set -e
#
# run at root of ascent repo
#
date
# run spack install, this will install ascent@develop
export ASCENT_VERSION=0.5.2-pre
export BASE_DIR=$WORLDWORK/csc340/software/ascent
export DEST_DIR=$BASE_DIR/${ASCENT_VERSION}/summit/openmp/gnu
mkdir -p $DEST_DIR
python scripts/uberenv/uberenv.py --spec="%gcc" \
       --pull \
       --install \
       --spack-config-dir="scripts/uberenv/spack_configs/olcf/summit_openmp/" \
       --prefix=${DEST_DIR}

# gen symlinks to important deps
python scripts/spack_install/gen_extra_install_symlinks.py ${DEST_DIR} cmake python conduit
# gen env helper script
rm public_env.sh
python scripts/spack_install/gen_public_install_env_script.py ${DEST_DIR} gcc/6.4.0
chmod a+x public_env.sh
cp public_env.sh $BASE_DIR/${ASCENT_VERSION}/summit/ascent_summit_setup_env_gcc_openmp.sh
# change perms to group write
chgrp -R csc340 $BASE_DIR/${ASCENT_VERSION}
chmod g+rwX -R $BASE_DIR/${ASCENT_VERSION}
# this space is already world readable, no need to change world perms
date