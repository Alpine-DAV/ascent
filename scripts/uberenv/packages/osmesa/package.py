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
from os.path import join as pjoin

class Osmesa(Package):
    """Mesa is an open-source implementation of the OpenGL 
    specification - a system for rendering interactive 3D graphics."""

    homepage = "http://www.mesa3d.org"
    url      = "http://www.example.com"

    version('7.5.2', '0f76124a7bb14a836bc95e8a28a6acf4')


    # patches for 7.10.2 from build_visit
    #patch("bv_osmesa_patch_1.patch",level=1) 
    #patch("bv_osmesa_patch_2.patch",level=1) 
    #patch("bv_osmesa_patch_3.patch",level=1) 

    def url_for_version(self, version):
        osmesa_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        osmesa_tar_path = pjoin(osmesa_tar_path,"MesaLib-7.5.2-OSMesa.tar.gz")
        url      = "file://" + osmesa_tar_path
        return url

    def install(self, spec, prefix):
        make()
        install_tree("lib",prefix.lib)
        install_tree("include",prefix.include)
        
        #configure("--prefix=%s" % prefix,
        # "--without-demos",
        #         "--with-driver=osmesa",
        #         "--disable-gallium",
        #         "--with-max-width=16384",
        #         "--with-max-height=16384",
        #         "--enable-glx-tls",
        #         "--disable-glu",
        #         "--disable-glw",
        #         "--disable-egl",
        #         "--disable-shared")
        #make()
        #make("install")

