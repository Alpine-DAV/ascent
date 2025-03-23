###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

from spack.package import *

import socket
import os
import platform
from os.path import join as pjoin

import spack.pkg.builtin.ascent

class UberenvAscent(spack.pkg.builtin.ascent.Ascent):
    """Spack Based Uberenv Build for Ascent Thirdparty Libs """

    homepage = "https://github.com/alpine-DAV/ascent"

    # default to building python when using uberenv
    variant("python",
            default=True,
            description="Build Python Support")

    # default to building docs when using uberenv
    variant("doc",
           default=True,
           description="Build deps needed to build Docs")

    depends_on("py-sphinx", when="+python+doc", type=("build","run"))
    depends_on("py-sphinx-rtd-theme", when="+python+doc", type=("build","run"))
    depends_on("py-sphinxcontrib-jquery", when="+python+doc", type=("build","run"))

    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-ascent.tar.gz")
        url      = "file://" + dummy_tar_path
        return url

    def hostconfig(self,spec,prefix):
        spack.pkg.builtin.ascent.Ascent.hostconfig(self)

    ###################################
    # build phases used by this package
    ###################################
    phases = ['hostconfig']
