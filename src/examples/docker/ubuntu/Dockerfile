# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

FROM alpinedav/ascent-ci:ubuntu-18-devel

# nodejs is needed by jupyterlab for widgets
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs
RUN rm -rf /var/lib/apt/lists/*

# add our 'user' with password 'docker'
RUN useradd -ms /bin/bash -G sudo user && echo "user:docker" | chpasswd
# allow sudo w/o password
RUN echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
 chmod 0440 /etc/sudoers.d/user
     
# run the rest as user
USER user
WORKDIR /home/user

# obtain a copy of ascent source from host env
COPY ascent.docker.src.tar.gz /home/user
RUN cd /home/user && tar -xzf ascent.docker.src.tar.gz

# obtain a copy of ascent source from host env,
# which we use to call uberenv
COPY ascent.docker.src.tar.gz /home/user/
# extract 
RUN cd /home/user && tar -xzf ascent.docker.src.tar.gz

# copy spack build script in
COPY docker_uberenv_build.sh /home/user/
#RUN cd /home/user/ && chmod -R a+x *.sh
RUN cd /home/user/ && ./docker_uberenv_build.sh


# install jupyter, matplotlib, cinema, etc via pip
RUN /home/user/uberenv_libs/spack/opt/spack/*/*/python*/bin/pip \
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
RUN /home/user/uberenv_libs/spack/opt/spack/*/*/python*/bin/jupyter \
    labextension install @jupyter-widgets/jupyterlab-manager

# configure a debug build with cmake
RUN cd /home/user/ascent && mkdir build-debug
RUN cd /home/user/ascent/build-debug && \
    /home/user/uberenv_libs/spack/opt/spack/*/*/cmake*/bin/cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=/home/user/ascent/install-debug \
    -C /home/user/uberenv_libs/*.cmake \
    ../src

# build, test, and install ascent
RUN cd /home/user/ascent/build-debug && make
#RUN cd /home/user/ascent/build-debug && env CTEST_OUTPUT_ON_FAILURE=1 make test
RUN cd /home/user/ascent/build-debug && make install

# install ascent bridge kernel into jupyter
RUN cd /home/user/ascent/install-debug/share/ascent/ascent_jupyter_bridge/ && \
    /home/user/uberenv_libs/spack/opt/spack/*/*/python*/bin/pip \
    install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt

RUN cd /home/user/ascent/install-debug/share/ascent/ascent_jupyter_bridge/ && \
    /home/user/uberenv_libs/spack/opt/spack/*/*/python*/bin/pip \
    install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    .

# setup ssh keys for passwordless ssh to localhost
RUN mkdir -p ~/.ssh
RUN ssh-keygen -b 2048 -t rsa -f ~/.ssh/id_rsa -q -N ""
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
RUN chmod 0600 ~/.ssh/authorized_keys

# setup jupyter password
RUN mkdir /home/user/.jupyter/
# the password is:
#  learn
RUN echo "c.NotebookApp.password = 'sha1:9777986fd066:283f673e1a311e2d5ef58c174eaebf3e1cb536dd'" > /home/user/.jupyter/jupyter_notebook_config.py
# set bash as default shell
RUN echo "c.NotebookApp.terminado_settings = { 'shell_command': ['/bin/bash'] }" >> /home/user/.jupyter/jupyter_notebook_config.py

# gen env script that points to spack installs of tpls
cd /home/user/ && python ascent/scripts/gen_spack_env_script.py cmake mpi python
RUN cp /home/user/s_env.sh ascent_docker_setup_env.sh
RUN echo "export PYTHONPATH=/home/user/ascent/install-debug/python-modules/" >> ascent_docker_setup_env.sh

# gen script that allows us to easily run jupyter
RUN cp /home/user/ascent_docker_setup_env.sh ascent_docker_run_jupyter.sh
RUN echo "jupyter notebook --ip=\"0.0.0.0\" --no-browser" >> ascent_docker_run_jupyter.sh

# gen script that allows us to easily run jupyter lab
RUN cp /home/user/ascent_docker_setup_env.sh ascent_docker_run_jupyterlab.sh
RUN echo "jupyter lab --ip=\"0.0.0.0\" --no-browser" >> ascent_docker_run_jupyterlab.sh

#make sure helpers are executable
RUN chmod 777 /home/user/ascent_docker_setup_env.sh
RUN chmod 777 /home/user/ascent_docker_run_jupyter.sh 
RUN chmod 777 /home/user/ascent_docker_run_jupyterlab.sh
RUN chmod 777 /home/user/ascent/install-debug/examples/ascent/tutorial/ascent_intro/*/cleanup.sh

# open port 8000, for use as http server
EXPOSE 8000
# open port 8888, for use by jupyter notebook http server
EXPOSE 8888
# open port 9000, for use by conduit relay
EXPOSE 9000
# open port 10000 for other apps (browse fs via http, etc)
EXPOSE 10000

CMD sudo service ssh start && /bin/bash
