# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

FROM rocm/dev-ubuntu-22.04:6.3.4

# add sudo to base cuda devel env
# so we can install additional packages as
# non-root, but admin default user on azure pipelines
RUN apt-get update && apt-get -y install sudo
# tzdata install wants to ask questions, so handled as sep case
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
# install std packages we need for cuda dev env and test
RUN apt-get update && apt-get -y install \
             binutils \
             python3 \
             python3-dev \
             libpython3-dev \
             python-is-python3 \
             perl \
             git \
             git-lfs \
             curl \
             libcurl4-openssl-dev \
             wget \
             tar \
             unzip \
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
             rocprim-dev \
             vim

