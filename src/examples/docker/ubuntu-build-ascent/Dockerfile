# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

#
# all in one build of ascent + jupyter container
#

FROM ubuntu:22.04
# add sudo to base ubuntu container
# so we can install additional packages as
# non-root, but admin default user on azure pipelines
RUN apt-get update && apt-get -y install sudo
# tzdata install wants to ask questions, so handled as sep case
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
# install std packages
RUN apt-get update && apt-get -y install \
             binutils \
             cmake \
             gcc \
             g++ \
             gfortran \
             python3 \
             python3-dev \
             python-is-python3 \
             python3.10-venv \
             clang \
             llvm \
             perl \
             git \
             git-lfs \
             curl \
             wget \
             tar \
             unzip \
             file \
             build-essential \
             libncurses-dev \
             libssl-dev \
             libblas-dev \
             liblapack-dev \
             liblapacke-dev \
             zlib1g-dev \
             libgdbm-dev \
             libreadline-dev \
             libsqlite3-dev \
             libbz2-dev \
             mpich \
             libmpich-dev \
             openssh-server \
             vim

# nodejs is needed by jupyterlab for widgets
RUN curl -sL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y nodejs
RUN rm -rf /var/lib/apt/lists/*

# obtain a copy of ascent source from host env and extract
COPY ascent.docker.src.tar.gz /
RUN tar -xzf ascent.docker.src.tar.gz

# change perms of source scripts
RUN chmod -R a+rwX /ascent/
RUN chmod +x /ascent/scripts/build_ascent/build_ascent.sh

# copy docker specific build script in
COPY docker_build_ascent.sh /
RUN  ./docker_build_ascent.sh

# change perms of build results
RUN chmod a+rwX /*.sh
RUN chmod -R a+rwX /install/

# add setup for login
RUN cp /ascent_docker_setup_env.sh /etc/profile.d/ascent_docker_setup_env.sh

# force the use of a login shell
RUN /bin/echo -e '#!/bin/bash -l\n' \
                 'exec "$@"\n' \
                > /etc/entrypoint.sh && \
        chmod a+x /etc/entrypoint.sh

# set main
ENTRYPOINT ["/etc/entrypoint.sh"]

# add our `user` with password `docker`
RUN useradd -ms /bin/bash -G sudo user && echo "user:docker" | chpasswd
# allow sudo w/o password
RUN echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
chmod 0440 /etc/sudoers.d/user
     
# run the rest as `user`
USER user
WORKDIR /home/user

# setup ssh keys for passwordless ssh to localhost
RUN mkdir -p ~/.ssh
RUN ssh-keygen -b 2048 -t rsa -f ~/.ssh/id_rsa -q -N ""
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
RUN chmod 0600 ~/.ssh/authorized_keys

# setup jupyter password
# the password is:
#  learn
RUN mkdir /home/user/.jupyter/
RUN echo "c.NotebookApp.password = 'sha1:9777986fd066:283f673e1a311e2d5ef58c174eaebf3e1cb536dd'" > /home/user/.jupyter/jupyter_notebook_config.py

# set bash as default jupyter shell
RUN echo "c.NotebookApp.terminado_settings = { 'shell_command': ['/bin/bash'] }" >> /home/user/.jupyter/jupyter_notebook_config.py

# gen script that allows us to easily run jupyter lab
RUN cp /ascent_docker_setup_env.sh /home/user/ascent_docker_run_jupyterlab.sh
RUN echo "jupyter lab --ip=\"0.0.0.0\" --no-browser" >> /home/user/ascent_docker_run_jupyterlab.sh

# symlink proxies under the notebooks dir for easy access in jupyter env
#RUN ln -s /ascent/install/examples/ascent/proxies/ /ascent/install/examples/ascent/tutorial/ascent_intro/notebooks/

# make sure helpers are executable
RUN sudo chmod 777 /home/user/ascent_docker_run_jupyterlab.sh
RUN sudo chmod 777 /ascent/install/examples/ascent/tutorial/ascent_intro/*/cleanup.sh

# open port 8888, for use by jupyter notebook http server
EXPOSE 8888

# default setup -- launch jupyter lab
CMD sudo service ssh start && cd /ascent/install/examples/ascent/tutorial/ascent_intro/notebooks/ && /home/user/ascent_docker_run_jupyterlab.sh
