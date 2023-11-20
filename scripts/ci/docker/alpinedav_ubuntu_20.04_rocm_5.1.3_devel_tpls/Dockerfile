# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

ARG ARCH_TAG="x86_64"
FROM alpinedav/ascent-devel:ubuntu-20.04-rocm-5.1.3-${ARCH_TAG}

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
