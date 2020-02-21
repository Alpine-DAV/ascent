FROM ubuntu:18.04
MAINTAINER Dan Lipsa <dan.lipsa@kitware.com>

RUN apt-get update
RUN apt-get install -y apt-utils bc imagemagick git curl unzip vim
RUN apt-get install -y python3 gcc g++ gfortran make cmake ninja-build
WORKDIR /root
RUN git clone https://github.com/spack/spack.git
RUN mkdir tests

CMD ["/bin/bash"]
