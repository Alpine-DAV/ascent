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
from spack.hooks.sbang import filter_shebang

class PySphinx(Package):
    """Sphinx Documentation Generator."""

    homepage = "http://sphinx-doc.org/"
    url      = "https://pypi.python.org/packages/8b/78/eeea2b837f911cdc301f5f05163f9729a2381cadd03ccf35b25afe816c90/Sphinx-1.4.5.tar.gz"

    version('1.4.5', '5c2cd2dac45dfa6123d067e32a89e89a')

    extends('python')
    
    depends_on("py-setuptools")
    depends_on("py-six")
    depends_on("py-jinja2")
    depends_on("py-pygments")
    depends_on("py-docutils")
    depends_on("py-imagesize")
    depends_on("py-snowballstemmer")
    depends_on("py-alabaster")
    depends_on("py-babel")
    depends_on("py-sphinx-rtd-theme")
    
    

    def install(self, spec, prefix):
        # simply install to the spack python
        python('setup.py', 'install') #, '--prefix=%s' % prefix)
        # sphinx_build lives in python's bin dir
        sphinx_scripts = ["sphinx-apidoc",
                          "sphinx-autogen",
                          "sphinx-build",
                          "sphinx-quickstart"]
        for script in sphinx_scripts:
            script_path = join_path(spec["python"].prefix,"bin",script)
            # use spack sbang to fix issues with shebang that is too long
            filter_shebang(script_path)



