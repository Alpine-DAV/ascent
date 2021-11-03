# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

FROM alpinedav/ascent-ci:ubuntu-18-devel

# nodejs is needed by jupyterlab for widgets
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs
RUN rm -rf /var/lib/apt/lists/*

# obtain a copy of ascent source from host env,
# which we use to call uberenv
COPY ascent.docker.src.tar.gz /
# extract 
RUN tar -xzf ascent.docker.src.tar.gz

# copy spack build script in
COPY docker_uberenv_build.sh /
RUN  ./docker_uberenv_build.sh

# install jupyter, matplotlib, cinema, etc via pip
RUN /uberenv_libs/spack/opt/spack/*/*/python*/bin/pip \
    install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    bash_kernel \
    jupyter \
    jupyterlab \
    matplotlib \
    pyyaml \
    cinemasci \
    scipy \
    scikit-learn \
    ipympl

# finish jupyter lab setup
RUN /uberenv_libs/spack/opt/spack/*/*/python*/bin/jupyter \
    labextension install @jupyter-widgets/jupyterlab-manager

# configure a debug build with cmake
RUN cd /ascent && mkdir build-debug 
RUN cd /ascent/build-debug && \
    /uberenv_libs/spack/opt/spack/*/*/cmake*/bin/cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=/ascent/install-debug \
    -DENABLE_BABELFLOW=OFF \
    -DADIOS2_DIR=IGNORE \
    -DFIDES_DIR=IGNORE \
    -C /uberenv_libs/*.cmake \
    ../src

# build, test, and install ascent
RUN cd /ascent/build-debug && make
# RUN cd /ascent/build-debug && env CTEST_OUTPUT_ON_FAILURE=1 make test
RUN cd /ascent/build-debug && make install


# install ascent bridge kernel into jupyter
RUN cd /ascent/install-debug/share/ascent/ascent_jupyter_bridge/ && \
    /uberenv_libs/spack/opt/spack/*/*/python*/bin/pip \
    install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt

RUN cd /ascent/install-debug/share/ascent/ascent_jupyter_bridge/ && \
    /uberenv_libs/spack/opt/spack/*/*/python*/bin/pip \
    install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    .

RUN chmod a+rwX /*.sh
RUN chmod -R a+rwX /uberenv_libs
RUN chmod -R a+rwX /ascent/

# add setup for login
RUN cp /ascent_docker_setup_env.sh /etc/profile.d/ascent_docker_setup_env.sh

# force the use of a login shell
RUN /bin/echo -e '#!/bin/bash -l\n' \
                 'exec "$@"\n' \
                > /etc/entrypoint.sh && \
        chmod a+x /etc/entrypoint.sh

ENTRYPOINT ["/etc/entrypoint.sh"]
CMD ["/bin/bash"]
