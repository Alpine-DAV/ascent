###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Strawman. 
# 
# For details, see: http://software.llnl.gov/strawman/.
# 
# Please also read strawman/LICENSE
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

import glob
import os
import platform

class Eavl(Package):
    homepage = "http://ft.ornl.gov/eavl/"
    url      = "https://github.com/jsmeredith/EAVL"

    version('github', 
            git='https://github.com/jsmeredith/EAVL.git',
            branch="rayTracer")

    # would like to use these in the future, but we need global variant support
    #variant('cuda',   default=False, description="Enable CUDA support.")
    #variant('openmp', default=False, description="Enable OpenMP support.")

    def install(self, spec, prefix):
        patched = open("configure").read().replace('NVCXXFLAGS="$NVCXXFLAGS -Xcompiler',
                                                   'NVCXXFLAGS="$NVCXXFLAGS -ccbin=$CXX -Xcompiler ')
        open("configure","w").write(patched)

        mpicc = which("mpicc")
        
        cfg_opts = ["--prefix=%s" % prefix,
                    "CXXFLAGS=-O3"]

        nvcc = which("nvcc")
        if not nvcc is None and "CUDA_PATH" in env.keys():
            cfg_opts.append("--with-cuda=%s" % env["CUDA_PATH"])
            # this fixes a problem w/ env vars at LLNL:
            if "CUDA_LIBS" in env.keys():
                cuda_libs = env["CUDA_LIBS"]
                if os.path.isdir(cuda_libs):
                    #clear cuda libs, Eavl expects them to be the actual lib names
                    env["CUDA_LIBS"] = ""

        if spec.satisfies('%intel'):
            cfg_opts.append("--with-openmp")

        env["CC"]  = env["SPACK_CC"]
        env["CXX"] = env["SPACK_CXX"]
        configure(*cfg_opts)
        make()
        mkdirp(prefix.lib)
        mkdirp(prefix.include)
        # copy libeavl.a
        install('lib/libeavl.a', prefix.lib)
        
        # the bvh headers is a bit diff than other headers
        install_tree('src/raytracing/bvh', join_path(prefix.include,"bvh"))
        
        header_globs = ["config/*.h",
                        "src/common/*.h",
                        'src/filters/*.h',
                        'src/math/*.h',
                        'src/rendering/*.h',
                        'src/raytracing/*.h',
                        'src/fonts/*.h',
                        'src/exporters/*.h',
                        'src/importers/*.h',
                        'src/operations/*.h']
        # copy headers
        for h_pattern in header_globs:
            header_files = glob.glob(h_pattern)
            for header_file in header_files:
                install(header_file,prefix.include)
        
  
