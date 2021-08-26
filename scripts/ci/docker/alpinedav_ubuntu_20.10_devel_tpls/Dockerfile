###############################################################################
# Copyright (c) 2015-2021, Lawrence Livermore National Security, LLC.
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

FROM alpinedav/ascent-ci:ubuntu-20.10-devel

# obtain a copy of ascent source from host env,
# which we use to call uberenv
COPY ascent.docker.src.tar.gz /
# extract ou
RUN tar -xzf ascent.docker.src.tar.gz

# at some sites ssl certs are intercepted, which cases issues fetching
# tpl sources via https

# to resolve this, either:
# 1) pass the "-k" option to uberenv (recommended),
# 2) install the proper certs into the image, or
# 3) use  the following commands to disable ssl for git and
#    curl (both are used by spack):
#RUN git config --global http.sslVerify false
#RUN echo insecure >> ~/.curlrc

#
# TODO re-enable dray
#

# bootstrap third party libs using spack and uberenv
RUN cd ascent && python scripts/uberenv/uberenv.py \
       -k \
       --spec "%gcc+mpi+python+babelflow+fides+adios2" \
       --spack-config-dir=scripts/uberenv_configs/spack_configs/configs/alpinedav/ubuntu_18_devel/ \
       --prefix=/uberenv_libs

# cleanup the spack build stuff to free up space
RUN uberenv_libs/spack/bin/spack clean --all

# change perms
RUN chmod -R a+rX /uberenv_libs

# create some helper scripts
RUN python ascent/scripts/gen_spack_env_script.py cmake mpi python
RUN echo "git clone --recursive https://github.com/Alpine-DAV/ascent.git" > clone.sh
RUN chmod +x clone.sh

# delete copy of source from host (ci will fetch new from repo)
RUN rm -rf ascent
