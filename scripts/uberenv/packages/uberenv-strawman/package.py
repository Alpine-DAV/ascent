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

import socket
import os
import platform
from os.path import join as pjoin


def cmake_cache_entry(name,value):
    return 'set("%s" "%s" CACHE PATH "")\n\n' % (name,value)
        


class UberenvStrawman(Package):
    """Spack Based Uberenv Build for Strawman Thirdparty Libs """

    homepage = "http://example.com"

    version('0.1', '8d378ef62dedc2df5db447b029b71200')
    
    # would like to use these in the future, but we need global variant support
    #variant('cuda',   default=False, description="Enable CUDA support.")
    #variant('openmp', default=False, description="Enable OpenMP support.")

    variant("doc",default=True,description="build third party dependencies for creating Strawman's docs")
    variant("python",default=True,description="build python 2")
    variant("mpich",default=False,description="build mpich as MPI lib for Strawman")
    
    depends_on("cmake@3.3.1")
    
    depends_on("icet")
    depends_on("vtkm")
    
    # python2
    depends_on("python", when="+python")
    depends_on("py-numpy", when="+python")
    
    
    depends_on("python", when="+doc")
    depends_on("py-sphinx", when="+doc")
    depends_on("py-breathe", when="+doc")
    

    #on osx, always build mpich for mpi support
    if "darwin" in platform.system().lower():
        depends_on("mpich")
        depends_on("conduit~silo~python3+mpich")
    else: # else, defer to the variant
        depends_on("conduit~silo~python3")
        depends_on("mpich",when="+mpich")
        depends_on("icet+mpich", when="+mpich")
        depends_on("conduit~silo~python3+mpich", when="+mpich")


    def url_for_version(self, version):
        dummy_tar_path =  os.path.abspath(pjoin(os.path.split(__file__)[0]))
        dummy_tar_path = pjoin(dummy_tar_path,"uberenv-strawman.tar.gz")
        url      = "file://" + dummy_tar_path
        return url
        
    def install(self, spec, prefix):
        dest_dir     = env["SPACK_DEBUG_LOG_DIR"]

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

        print "cmake executable: %s" % cmake_exe
        
        #######################
        # Check for MPI
        #######################
        mpicc   = which("mpicc")
        mpicxx  = which("mpicxx")
        mpif90  = which("mpif90")
        mpiexec = which("mpiexec")
        nvcc    = which("nvcc")
        
        if not nvcc is None:
            enable_cuda = "ON"
        else:
            enable_cuda = "OFF"

        if env.has_key("UBERENV_OPENMP") or spec.satisfies('%intel'):
            enable_openmp = "ON"
        else:
            enable_openmp = "OFF"


        #######################
        # Create host-config
        #######################
        host_cfg_fname = "%s-%s-%s.cmake" % (socket.gethostname(),sys_type,spec.compiler)
        host_cfg_fname = pjoin(dest_dir,host_cfg_fname)

        cfg = open(host_cfg_fname,"w")
        cfg.write("##################################\n")
        cfg.write("# uberenv host-config\n")
        cfg.write("##################################\n")
        cfg.write("# %s-%s\n" % (sys_type,spec.compiler))
        cfg.write("##################################\n\n")
        # show path to cmake for reference
        cfg.write("# cmake from uberenv\n")
        if cmake_exe is None:
            cfg.write("# cmake not built by uberenv\n\n");
        else:
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
        # python
        #######################
        if "+python" in spec:
            python_exe = pjoin(spec['python'].prefix.bin,"python")
            cfg.write("# Enable python module builds\n")
            cfg.write(cmake_cache_entry("ENABLE_PYTHON","ON"))
            cfg.write("# python from uberenv\n")
            cfg.write(cmake_cache_entry("PYTHON_EXECUTABLE",python_exe))

        #######################
        # sphinx
        #######################
        sphinx_build_exe = pjoin(spec['python'].prefix.bin,"sphinx-build")
        cfg.write("# python from uberenv\n")
        cfg.write(cmake_cache_entry("SPHINX_EXECUTABLE",sphinx_build_exe))


        #######################
        # sphinx
        #######################
        if "+doc" in spec:
            sphinx_build_exe = pjoin(spec['python'].prefix.bin,"sphinx-build")
            cfg.write("# sphinx from uberenv\n")
            cfg.write(cmake_cache_entry("SPHINX_EXECUTABLE",sphinx_build_exe))

        #######################
        # openmp
        #######################
        cfg.write("# OPENMP Support\n")   
        cfg.write(cmake_cache_entry("ENABLE_OPENMP",enable_openmp));

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
        # cuda
        #######################
        cfg.write("# CUDA support\n")
        cfg.write(cmake_cache_entry('ENABLE_CUDA',enable_cuda));
        if not nvcc is None:
            cfg.write(cmake_cache_entry('CUDA_BIN_DIR',os.path.dirname(nvcc.command)))

        #######################
        # conduit
        #######################
        cfg.write("# conduit from uberenv\n")
        cfg.write(cmake_cache_entry("CONDUIT_DIR", spec['conduit'].prefix))
        cfg.write("\n")

        #######################
        # hdf5
        #######################
        cfg.write("# hdf5 from uberenv\n")
        cfg.write(cmake_cache_entry("HDF5_DIR", spec['hdf5'].prefix))
        cfg.write("\n")

        #######################
        # icet
        #######################
        cfg.write("# icet from uberenv\n")
        cfg.write(cmake_cache_entry("ICET_DIR", spec['icet'].prefix))
        cfg.write("\n")

        #######################
        # eavl + tpls
        #######################

        cfg.write("\n# eval support\n\n")
        cfg.write("# osmesa from uberenv\n")
        cfg.write(cmake_cache_entry("OSMESA_DIR", spec['osmesa'].prefix))
        cfg.write("# eavl from uberenv\n")
        cfg.write(cmake_cache_entry("EAVL_DIR", spec['eavl'].prefix))
        cfg.write("\n")

        #######################
        # vtkm + tpls
        #######################

        cfg.write("\n# vtkm support\n\n")    

        cfg.write("# tbb from uberenv\n")
        cfg.write(cmake_cache_entry("TBB_DIR", spec['tbb'].prefix))

        cfg.write("# vtkm from uberenv\n")
        cfg.write(cmake_cache_entry("VTKM_DIR", spec['vtkm'].prefix))


        cfg.write("##################################\n")
        cfg.write("# end uberenv host-config\n")
        cfg.write("##################################\n")

        cfg.close()
        mkdirp(prefix)
        install(host_cfg_fname,prefix)
        print "[result host-config file: %s]" % host_cfg_fname

        
