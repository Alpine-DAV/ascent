# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

FROM nvidia/cuda:10.1-devel-ubuntu18.04
# add sudo to base cuda devel env
# so we can install additional packages as
# non-root, but admin default user on azure pipelines
RUN apt-get update && apt-get -y install sudo
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
               unzip \
               vim


