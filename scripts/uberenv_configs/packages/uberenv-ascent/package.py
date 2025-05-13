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

############
# if you don't have a custom ascent package we need to use:
# from spack.pkg.builtin.ascent import Ascent

from spack.pkg.ascent_uberenv_repo.ascent import Ascent

class UberenvAscent( Ascent ):
    """Spack Based Uberenv Build for Ascent Thirdparty Libs """

    homepage = "https://github.com/alpine-DAV/ascent"

    version("develop",
            sha256="21d3663781975432144037270698d493a7f8fa876ede7da51618335be468168f",
            preferred=True)

    # default to building python when using uberenv
    variant("python",
            default=True,
            description="Build Python Support")

    # default to building docs when using uberenv
    variant("doc",
           default=True,
           description="Build deps needed to build Docs")
    # default to building caliper when using uberenv
    variant("caliper", default=True, description="Build Caliper support")


    depends_on("py-sphinx", when="+python+doc", type=("build","run"))
    depends_on("py-sphinx-rtd-theme", when="+python+doc", type=("build","run"))
    depends_on("py-sphinxcontrib-jquery", when="+python+doc", type=("build","run"))
    depends_on("py-setuptools", when="+python", type=("build", "run"))
    depends_on("py-wheel", when="+python", type=("build", "run"))

    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-ascent.tar.gz")
        url      = "file://" + dummy_tar_path
        return url

    def hostconfig(self,spec,prefix):
         Ascent.hostconfig(self)

    ###################################
    # build phases used by this package
    ###################################
    phases = ['hostconfig']
