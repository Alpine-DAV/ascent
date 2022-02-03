#!/bin/bash
set -e
#
# run at root of ascent repo
#
date
# run spack install, this will install ascent@develop
export ASCENT_VERSION=0.8.0
#export BASE_DIR=/gpfs/alpine/proj-shared/aph114/ascent/
#export DEST_DIR=$BASE_DIR/${ASCENT_VERSION}/summit/cuda/gnu
export BASE_DIR=/ccs/home/cyrush/WORKSCRATCH
export DEST_DIR=$BASE_DIR/ASCENT_INSTALL/{$ASCENT_VERSION}
mkdir -p $DEST_DIR
python3 scripts/uberenv/uberenv.py --spec="%gcc" \
       --pull \
       --install \
       --spack-config-dir="scripts/uberenv_configs/spack_configs/configs/olcf/summit_gcc_9.3.0_warpx/" \
       --prefix=${DEST_DIR}

# gen symlinks to important deps
python3 scripts/spack_install/gen_extra_install_symlinks.py ${DEST_DIR} cmake conduit
# gen env helper script
rm -f public_env.sh
python3 scripts/spack_install/gen_public_install_env_script.py ${DEST_DIR} gcc/9.3.0 cuda
chmod a+x public_env.sh
cp public_env.sh $BASE_DIR/${ASCENT_VERSION}/summit/ascent_summit_setup_env_gcc_cuda.sh
# change perms to group write
chgrp -R aph114 $BASE_DIR/${ASCENT_VERSION}
chmod g+rwX -R $BASE_DIR/${ASCENT_VERSION}
# world shared no longer means world shared by default, so lets change perms for all
chmod a+rX -R ${BASE_DIR}/${ASCENT_VERSION}/
date
