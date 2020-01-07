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

from spack import *

import socket
import os
import platform
from os.path import join as pjoin

from .ascent import Ascent

class UberenvAscent(Ascent):
    """Spack Based Uberenv Build for Ascent Thirdparty Libs """

    homepage = "https://github.com/alpine-DAV/ascent"

    version('0.5.0', '21d3663781975432144037270698d493a7f8fa876ede7da51618335be468168f')

    # default to building docs when using uberenv
    variant("doc",
            default=True,
            description="Build deps needed to create Conduit's Docs")

    variant("babelflow", 
            default=False,
            description="Build with BabelFlow")

    variant("pmt", 
            default=False,
            description="Build with ParallelMergeTree analysis")

    depends_on('babelflow@develop', when='+babelflow')
    depends_on('pmt@develop', when='+babelflow')

    # in upstream spack package
    depends_on("cmake@3.14.1:3.14.5", when="+cmake")

    def cmake_args(self):
        cmake_args=""
        if '+babelflow' in self.spec:
            cmake_args.extend([
                '-DENABLE_BABELFLOW=ON'
            ])
            cmake(*cmake_args)

    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-ascent.tar.gz")
        url      = "file://" + dummy_tar_path
        return url

    def install(self, spec, prefix):
        """
        Build and install Ascent.
        """
        with working_dir('spack-build', create=True):
            host_cfg_fname = self.create_host_config(spec, prefix)
            # place a copy in the spack install dir for the uberenv-conduit package
            mkdirp(prefix)
            install(host_cfg_fname,prefix)
            install(host_cfg_fname,env["SPACK_DEBUG_LOG_DIR"])

            #######################
            # BABELFLOW
            #######################

            #if "+babelflow" in spec:
            #    host_cfg_fname.write(cmake_cache_entry("ENABLE_BABELFLOW", "ON"))
            #    host_cfg_fname.write(cmake_cache_entry("BabelFlow_DIR", spec['babelflow'].prefix))
            #    host_cfg_fname.write(cmake_cache_entry("PMT_DIR", spec['pmt'].prefix))
