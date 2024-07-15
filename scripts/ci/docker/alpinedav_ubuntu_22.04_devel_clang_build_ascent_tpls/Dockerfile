# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

ARG TAG_ARCH=x86_64
FROM alpinedav/ascent-devel:ubuntu-22.04-${TAG_ARCH}

ENV TAG_ARCH ${TAG_ARCH}

# obtain a copy of ascent source from host env,
# which we use to call uberenv
COPY ascent.docker.src.tar.gz /
# extract 
RUN tar -xzf ascent.docker.src.tar.gz

# copy spack build script in
COPY docker_uberenv_build.sh docker_env_setup.sh /
RUN chmod -R a+x /*.sh

RUN /docker_uberenv_build.sh

RUN /docker_env_setup.sh
