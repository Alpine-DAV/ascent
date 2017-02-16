###############################################################################
# Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see: http://software.llnl.gov/conduit/.
# 
# Please also read conduit/LICENSE
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


def cmake_cache_entry(name,value):
    return 'set(%s "%s" CACHE PATH "")\n\n' % (name,value)
        

class Conduit(Package):
    """Spack package for Conduit"""

    homepage = "http://software.llnl.gov/conduit"

    homepage = "http://software.llnl.gov/conduit/"
    url      = "https://github.com/LLNL/conduit/archive/v0.2.1.tar.gz"

    version('0.2.1', 'cd2b42c76f70ac3546582b6da77c6028')
    version('0.2.0', 'd595573dedf55514c11d7391092fd760')

    variant("hdf5",default=True,description="build third party dependencies for Conduit HDF5 support")
    variant("silo",default=True,description="build third party dependencies for Conduit Silo support")
    
    variant("doc",default=True,description="build third party dependencies for creating Conduit's docs")
    variant("python",default=True,description="build python 2")
    variant("python3",default=True,description="build python 3")
    variant("mpich",default=False,description="build mpich as MPI lib for Conduit")


    ###########################
    # standard spack packages
    ###########################
    
    ##########################
    # uberenv custom packages
    ##########################

    #on osx, always build mpich for mpi support
    if "darwin" in platform.system().lower():
        depends_on("mpich")
    else: # else, defer to the variant
        depends_on("mpich",when="+mpich")

    #######################
    # CMake
    #######################
    depends_on("cmake@3.4.0")
    
    #######################
    # python
    #######################

    # python2
    depends_on("python", when="+python")
    depends_on("py-numpy", when="+python")
    depends_on("py-sphinx", when="+python+doc")
    depends_on("py-breathe", when="+python+doc")

    # python3
    depends_on("python3", when="+python3")
    depends_on("py3-numpy",when="+python3")
    depends_on("py3-sphinx", when="+python3+doc")
    depends_on("py3-breathe", when="+python3+doc")

    #######################
    # i/o packages
    #######################
    depends_on("hdf5",when="+hdf5")
    depends_on("silo",when="+silo")

    def create_host_config(self,spec,prefix):
        c_compiler   = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]
        f_compiler   = None
    
        # see if we should enable fortran support
        if "SPACK_FC" in env.keys():
            # even if this is set, it may not exist
            # do one more sanity check
            if os.path.isfile(env["SPACK_FC"]):
                f_compiler  = env["SPACK_FC"]

        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if env.has_key("SYS_TYPE"):
            sys_type = env["SYS_TYPE"]
    
        #######################
        # TPL Paths
        #######################
        cmake_exe  = pjoin(spec['cmake'].prefix.bin,"cmake")
    
        #######################
        # Check for MPI
        #######################
        mpicc   = which("mpicc")
        mpicxx  = which("mpicxx")
        mpif90  = which("mpif90")
        mpiexec = which("mpiexec")
    
        print "cmake executable: %s" % cmake_exe
    
        #######################
        # Create host-config
        #######################
        host_cfg_fname = "%s-%s-%s.cmake" % (socket.gethostname(),sys_type,spec.compiler)

        cfg = open(host_cfg_fname,"w")
        cfg.write("##################################\n")
        cfg.write("# uberenv host-config\n")
        cfg.write("##################################\n")
        cfg.write("# %s-%s\n" % (sys_type,spec.compiler))
        cfg.write("##################################\n\n")
        # show path to cmake for reference
        cfg.write("# cmake from uberenv\n")
        cfg.write("# cmake exectuable path: %s\n\n" % cmake_exe)
    
        #######################
        #######################
        # compiler settings
        #######################
        #######################
    
        cfg.write("#######\n")
        cfg.write("# using %s compiler spec\n" % spec.compiler)
        cfg.write("#######\n\n")
        cfg.write("# c compiler used by spack\n")
        cfg.write(cmake_cache_entry("CMAKE_C_COMPILER",c_compiler))
        cfg.write("# cpp compiler used by spack\n")
        cfg.write(cmake_cache_entry("CMAKE_CXX_COMPILER",cpp_compiler))
    
        cfg.write("# fortran compiler used by spack\n")
        if not f_compiler is None:
            cfg.write(cmake_cache_entry("ENABLE_FORTRAN","ON"))
            cfg.write(cmake_cache_entry("CMAKE_Fortran_COMPILER",f_compiler))
        else:
            cfg.write("# no fortran compiler found\n\n")
            cfg.write(cmake_cache_entry("ENABLE_FORTRAN","OFF"))

        #######################
        #######################
        # python
        #######################
        cfg.write("# Python Support\n")
        #######################
        # python 2
        #######################
        if "+python" in spec:
            python_exe = pjoin(spec['python'].prefix.bin,"python")
            cfg.write("# Enable python module builds\n")
            cfg.write(cmake_cache_entry("ENABLE_PYTHON","ON"))
            cfg.write("# python from uberenv\n")
            cfg.write(cmake_cache_entry("PYTHON_EXECUTABLE",python_exe))
    
        if "+doc" in spec:
            sphinx_build_exe = pjoin(spec['python'].prefix.bin,"sphinx-build")
            cfg.write("# sphinx from uberenv\n")
            cfg.write(cmake_cache_entry("SPHINX_EXECUTABLE",sphinx_build_exe))

        #######################
        # python 3
        #######################
        if "+python3" in spec:
            python3_exe      = pjoin(spec['python3'].prefix.bin,"python3")
            cfg.write("# python3 from uberenv\n")
            cfg.write("#" + cmake_cache_entry("PYTHON_EXECUTABLE",python3_exe))
            if "+doc" in spec:
                py3_sphinx_build_exe = pjoin(spec['python3'].prefix.bin,"sphinx-build")
                cfg.write("# sphinx from uberenv\n")
                cfg.write("#" + cmake_cache_entry("SPHINX_EXECUTABLE",py3_sphinx_build_exe))

        if not "+python" in spec or "+pyhton3" in spec:
            cfg.write(cmake_cache_entry("ENABLE_PYTHON","OFF"))

        #######################
        # mpi
        #######################
        cfg.write("# MPI Support\n")
        if not mpicc is None:
            cfg.write(cmake_cache_entry("ENABLE_MPI","ON"))
            cfg.write(cmake_cache_entry("MPI_C_COMPILER",mpicc.command))
        else:
            cfg.write(cmake_cache_entry("ENABLE_MPI","OFF"))

        # we use `mpicc` as `MPI_CXX_COMPILER` b/c we don't want to introduce 
        # linking deps to the MPI C++ libs (we aren't using C++ features of MPI)
        if not mpicxx is None:
            cfg.write(cmake_cache_entry("MPI_CXX_COMPILER",mpicc.command))
        if not mpif90 is None:
            cfg.write(cmake_cache_entry("MPI_Fortran_COMPILER", mpif90.command))
        if not mpiexec is None:
            cfg.write(cmake_cache_entry("MPIEXEC", mpiexec.command))

        #######################
        #######################
        # i/o packages
        #######################
        #######################
        cfg.write("# I/O Packages\n\n")
        #######################
        # hdf5
        #######################
        cfg.write("# hdf5 from uberenv\n")
        if "+hdf5" in spec:
            cfg.write(cmake_cache_entry("HDF5_DIR", spec['hdf5'].prefix))
        else:
            cfg.write("# hdf5 not built by uberenv\n")
        #######################
        # silo
        #######################
        cfg.write("# silo from uberenv\n")
        if "+silo" in spec:
            cfg.write(cmake_cache_entry("SILO_DIR", spec['silo'].prefix))
        else:
            cfg.write("# silo not built by uberenv\n")

        cfg.write("##################################\n")
        cfg.write("# end uberenv host-config\n")
        cfg.write("##################################\n")
        cfg.close()
    
        # place a copy in the spack install dir for the conduit package 
        # mkdirp(prefix)
        # install(host_cfg_fname,prefix)
        # host_cfg_fname = pjoin(prefix,host_cfg_fname)
        host_cfg_fname = os.path.abspath(host_cfg_fname)
        print "[result host-config file: %s]" % host_cfg_fname
        return host_cfg_fname
    
    def install(self, spec, prefix):
        with working_dir('spack-build', create=True):
            host_cfg_fname = self.create_host_config(spec,prefix)
            cmake_args = []
            cmake_args.extend(std_cmake_args)
            cmake_args.extend(["-C", host_cfg_fname, "../src"])
            cmake(*cmake_args)
            make()
            make("install")


