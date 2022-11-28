###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
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

    # default to building python when using uberenv
    variant("python",
            default=True,
            description="Build Python Support")
    #
    # NOTE: THIS IS OFF DUE TO SPACK FETCH ISSUE WITH SPHINX
    # default to building docs when using uberenv
    #variant("doc",
    #        default=True,
    #        description="Build deps needed to build Docs")

    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-ascent.tar.gz")
        url      = "file://" + dummy_tar_path
        return url
    
    ###################################
    # build phases used by this package
    ###################################
    phases = ['hostconfig']
