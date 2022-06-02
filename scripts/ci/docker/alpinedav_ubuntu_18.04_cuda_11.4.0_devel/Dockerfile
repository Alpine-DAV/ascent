# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

FROM nvidia/cuda:11.4.0-devel-ubuntu18.04
# add sudo to base cuda devel env
# so we can install additional packages as
# non-root, but admin default user on azure pipelines
RUN apt-get update && apt-get -y install sudo
# tzdata install wants to ask questions, so handled as sep case
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
# install std packages we need for cuda dev env and test
RUN apt-get update && apt-get -y install \
               git \
               git-lfs \
               python \
               gfortran \
               zlib1g-dev \
               curl \
               mpich \
               libmpich-dev \
               libblas-dev \
               liblapack-dev \
               liblapacke-dev \
               libcublas10 \
               libcublas-dev \
               unzip \
               openssh-server \
               vim


