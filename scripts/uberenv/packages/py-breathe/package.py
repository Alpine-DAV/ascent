###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Ascent. 
# 
# For details, see: http://software.llnl.gov/ascent/.
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
from spack.hooks.sbang import filter_shebang

class PyBreathe(Package):
    """Breathe provides a bridge between the Sphinx and Doxygen documentation systems."""

    homepage = "https://breathe.readthedocs.org/en/latest/"
    url      = "https://pypi.python.org/packages/source/b/breathe/breathe-4.0.0.tar.gz"

    version('4.0.0', '32316d5a890a3124ea3e8a9e0b2b3b97')

    extends('python')


    def install(self, spec, prefix):
        # breathe + sphinx doesn't play well with --prefix installs, for now simply 
        # install to the spack python
        python('setup.py', 'install') #, '--prefix=%s' % prefix)
        # sphinx_build lives in python's bin dir and for some reason breathe's
        # setuptools install reinstalls the sphinx build scripts (even though
        # it finds the existing sphinx install).
        # To keep this from undermining us, we need to patch the new copies
        # of these scripts.
        sphinx_scripts = ["sphinx-apidoc",
                          "sphinx-autogen",
                          "sphinx-build",
                          "sphinx-quickstart"]
        for script in sphinx_scripts:
            script_path = join_path(spec["python"].prefix,"bin",script)
            # use spack sbang to fix issues with shebang that is too long
            filter_shebang(script_path)
