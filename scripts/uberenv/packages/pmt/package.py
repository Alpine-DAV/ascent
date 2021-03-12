# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

class Pmt(CMakePackage):
    """FIXME: Put a proper description of your package here."""

    homepage = "https://bitbucket.org/cedmav/parallelmergetree"
    url      = "https://bitbucket.org/cedmav/parallelmergetree/"

    maintainers = ['spetruzza']


    version('1.0.2',
            git='https://bitbucket.org/cedmav/parallelmergetree.git',
            tag='v1.0.2',
            submodules=True)

    depends_on('babelflow')

    variant("shared", default=True, description="Build ParallelMergeTree as shared libs")

    def cmake_args(self):
        args = []

        if "+shared" in self.spec:
            args.append('-DBUILD_SHARED_LIBS=ON')
        else:
            args.append('-DBUILD_SHARED_LIBS=OFF')

        args.append('-DLIBRARY_ONLY=ON')
        args.append('-DBabelFlow_DIR={0}'.format(
                    self.spec['babelflow'].prefix))

        return args
