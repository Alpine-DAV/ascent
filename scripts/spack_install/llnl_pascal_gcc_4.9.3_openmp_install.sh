#!/bin/bash
set -e
#
# run at root of ascent repo
#
date
# run spack install, this will install ascent@develop
export BASE_DIR=/usr/gapps/conduit/software/ascent
export ASCENT_VERSION=2020-11-06
export DEST_DIR=${BASE_DIR}/${ASCENT_VERSION}/toss_3_x86_64_ib/openmp/gnu
mkdir -p $DEST_DIR
python scripts/uberenv/uberenv.py --spec="%gcc+doc" \
       --pull \
       --install \
       --spack-config-dir="scripts/uberenv_configs/spack_configs/llnl/pascal_openmp/" \
       --prefix=${DEST_DIR}

# gen symlinks to important deps
python scripts/spack_install/gen_extra_install_symlinks.py ${DEST_DIR} cmake python conduit
# gen env helper script
rm -f public_env.sh
python scripts/spack_install/gen_public_install_env_script.py ${DEST_DIR} gcc/4.9.3
chmod a+x public_env.sh
cp public_env.sh ${BASE_DIR}/${ASCENT_VERSION}/toss_3_x86_64_ib/ascent_toss_3_x86_64_ib_setup_env_gcc_openmp.sh
# change perms to world readable
chmod a+rX -R /usr/gapps/conduit/software/ascent
date

