###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

FROM ubuntu:latest
MAINTAINER Cyrus Harrison <cyrush@llnl.gov>

# fetch build env
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    curl \
    build-essential \
    gcc \
    g++ \
    gfortran \
    zlib1g-dev \
    python \
    tar \
    unzip \
    perl \
    openmpi-bin \
    libopenmpi-dev \
    libncurses-dev \
    libssl-dev \
    libopenblas-dev \
    zlib1g-dev \
    libgdbm-dev \
    libreadline-dev \
    libsqlite3-dev \
    libbz2-dev \
    vim \
    emacs \
 && rm -rf /var/lib/apt/lists/*


RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user

# obtain a copy of ascent source from host env
COPY ascent.docker.src.tar.gz /home/user
RUN cd /home/user && tar -xzf ascent.docker.src.tar.gz

# at some sites ssl certs are intercepted, which cases issues fetching
# tpl sources via https

# to resolve this, either:
# 1) pass the "-k" option to uberenv (recommended),
# 2) install the proper certs into the image, or
# 3) use  the following commands to disable ssl for git and
#    curl (both are used by spack):
#RUN git config --global http.sslVerify false
#RUN echo insecure >> ~/.curlrc

# bootstrap third party libs using spack and uberenv
# for this example we use mpich for MPI b/c openmpi's mpiexec
# will not run for the root user, also avoid mpi4py after 2.0,
# since it reqs cython and cython's src tarball has perms set
# in a way that fails in docker
RUN cd /home/user/ascent && python scripts/uberenv/uberenv.py \
       -k \
       --spec "%gcc+mpi+mfem ^python@3.6.3" \
       --spack-config-dir=scripts/uberenv/spack_configs/docker/ubuntu/

# install jupyter and matplotlib via pip
RUN /home/user//ascent/uberenv_libs/spack/opt/spack/*/*/python*/bin/pip \
    install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    jupyter matplotlib

# configure a debug build with cmake
RUN cd /home/user/ascent && mkdir build-debug
RUN cd /home/user/ascent/build-debug && \
    ../uberenv_libs/spack/opt/spack/*/*/cmake*/bin/cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=/home/user/ascent/install-debug \
    -C ../uberenv_libs/*.cmake \
    ../src

# build, test, and install conduit
RUN cd /home/user/ascent/build-debug && make
RUN cd /home/user/ascent/build-debug && env CTEST_OUTPUT_ON_FAILURE=1 make test
RUN cd /home/user/ascent/build-debug && make install

# install ascent bridge kernel into jupyter
RUN cd /home/user/ascent/install-debug/share/ascent/ascent_jupyter_bridge/ && \
    /home/user/ascent/uberenv_libs/spack/opt/spack/*/*/python*/bin/pip \
    install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt

RUN cd /home/user/ascent/install-debug/share/ascent/ascent_jupyter_bridge/ && \
    /home/user/ascent/uberenv_libs/spack/opt/spack/*/*/python*/bin/pip \
    install \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    .

# gen env script that points to spack installs of tpls
RUN cd /home/user/ascent && python scripts/gen_spack_env_script.py cmake mpi python
RUN cp /home/user/ascent/s_env.sh ascent_docker_setup_env.sh
RUN cat "PYTHONPATH=/home/user/ascent/install-debug/python-modules/" >> ascent_docker_setup_env.sh

# setup docker password
RUN mkdir /home/user/.jupyter/
RUN echo "c.NotebookApp.password = 'sha1:d8baf2aa447b:aa60264fe892fc31c80c4650d49b1adeaa09944e'" > /home/user/.jupyter/jupyter_notebook_config.py

# gen script that allows us to easily run jupyter
RUN cp /home/user/ascent/ascent_docker_setup_env.sh ascent_docker_run_jupyter.sh
RUN echo "jupyter notebook --ip=\"0.0.0.0\" --no-browser" >> ascent_docker_run_jupyter.sh


# open port 8000, for use as http server
EXPOSE 8000
# open port 8888, for use by jupyter notebook http server
EXPOSE 8888
# open port 9000, for use by conduit relay
EXPOSE 9000
# open port 10000 for other apps (browse fs via http, etc)
EXPOSE 10000

CMD ["/bin/bash"]
