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
import os
import glob

import llnl.util.tty as tty

class Tbb(Package):
    """
    Intel's TBB is a widely used C++ template library for task parallelism.
    """
    homepage = "https://www.threadingbuildingblocks.org/"
    
    #
    # The TBB website has a bunch of guards to prevent downloads.
    # We keep our own copy in our tbb package directory. 
    #
    
    #url      = "http://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb44_20160128oss_src.tgz"
    url      = "http://www.example.com"

    version('4.4.3', '9d8a4cdf43496f1b3f7c473a5248e5cc')

    def url_for_version(self, version):
        tbb_tar_path =  os.path.abspath(join_path(os.path.dirname(__file__)))
        tbb_tar_path = join_path(tbb_tar_path,"tbb44_20160128oss_src.tgz")
        url      = "file://" + tbb_tar_path
        return url
    
    def coerce_to_spack(self,tbb_build_subdir):
        for compiler in ["icc","gcc","clang"]:
              fs = glob.glob(join_path(tbb_build_subdir,"*.%s.inc" % compiler ))
              for f in fs:
                  lines = open(f).readlines()
                  of = open(f,"w")
                  for l in lines:
                      if l.strip().startswith("CPLUS ="):
                        of.write("# coerced to spack\n")
                        of.write("CPLUS = $(CXX)\n")
                      elif l.strip().startswith("CPLUS ="):
                        of.write("# coerced to spack\n")
                        of.write("CONLY = $(CC)\n")
                      else:
                        of.write(l);

    def install(self, spec, prefix):
        #
        # we need to follow TBB's compiler selection logic to get the proper build + link flags
        # but we still need to use spack's compiler wrappers
        # to accomplish this, we do two things:
        #
        # * Look at the spack spec to determine which compiler we should pass to tbb's Makefile
        #
        # * patch tbb's build system to use the compiler wrappers (CC, CXX) for
        #    icc, gcc, clang
        #    (see coerce_to_spack())
        #

        self.coerce_to_spack("build")
        
        if spec.satisfies('%clang'):
            tbb_compiler = "clang"
        elif spec.satisfies('%intel'):
            tbb_compiler = "icc"
        else:
            tbb_compiler = "gcc"


        mkdirp(prefix)
        mkdirp(prefix.lib)        

        #
        # tbb does not have a configure script or make install target
        # we simply call make, and try to put the pieces together 
        #

        make("compiler=%s"  %(tbb_compiler))

        # Note, I tried try tbb_build_dir option, which quickly errored out ...""
        # make("compiler=%s tbb_build_dir=%s"  %(tbb_compiler,prefix.lib))

        
        # install headers to {prefix}/include
        install_tree('include',prefix.include)

        # install libs to {prefix}/lib
        tbb_lib_names = ["libtbb",
                         "libtbbmalloc",
                         "libtbbmalloc_proxy"]

        for lib_name in tbb_lib_names:
            # install release libs
            fs = glob.glob(join_path("build","*release",lib_name + ".*"))
            if len(fs) <1:
                tty.error("Failed to find release lib: " + lib_name)
            for f in fs:
                tty.info("installing:" + f);
                install(f, prefix.lib)
            # install debug libs if they exist
            fs = glob.glob(join_path("build","*debug",lib_name + "_debug.*"))
            for f in fs:
                tty.info("installing:" + f);
                install(f, prefix.lib)


  

