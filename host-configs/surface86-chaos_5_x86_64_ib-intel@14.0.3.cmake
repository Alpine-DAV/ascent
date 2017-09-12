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

##################################
# uberenv host-config
##################################
# chaos_5_x86_64_ib-intel@14.0.3
##################################

# cmake from uberenv
# cmake exectuable path: /usr/gapps/visit/ascent/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/intel-14.0.3/cmake-3.4.3-4yzlizfrgvnzilihovatbu6oppfuhrv3/bin/cmake

#######
# using intel@14.0.3 compiler spec
#######

# c compiler used by spack
set(CMAKE_C_COMPILER "/usr/local/bin/icc" CACHE PATH "")

# cpp compiler used by spack
set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc" CACHE PATH "")

# fortran compiler used by spack
set(CMAKE_Fortran_COMPILER  "/usr/local/bin/ifort" CACHE PATH "")

# OPENMP Support
set(ENABLE_OPENMP ON CACHE PATH "")

# MPI Support
set(ENABLE_MPI  ON CACHE PATH "")

set(MPI_C_COMPILER  "/usr/local/tools/mvapich2-intel-2.0/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/local/tools/mvapich2-intel-2.0/bin/mpicc" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/local/tools/mvapich2-intel-2.0/bin/mpif90" CACHE PATH "")

set(MPIEXEC /usr/bin/srun CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG -n CACHE PATH "")



# CUDA support
set(ENABLE_CUDA ON CACHE PATH "")
#set(ENABLE_CUDA OFF CACHE PATH "")

set(CUDA_BIN_DIR /opt/cudatoolkit-7.0/bin CACHE PATH "")

# sphinx from uberenv
# not built ...
# conduit from uberenv
set(CONDUIT_DIR "/nfs/tmp2/larsen30/conduit/install" CACHE PATH "")
#set(CONDUIT_DIR " /usr/gapps/visit/ascent/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/intel-14.0.3/conduit-github-naws5eho7jxgjaaldoubcarv5v3x4sgt/" CACHE PATH "")

# icet from uberenv
set(ICET_DIR "/usr/gapps/visit/strawman/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/intel-14.0.3/icet-icet-master-hmg4hcjrztyseukkalbvxaga353ys6es" CACHE PATH "")

#
# vtkm support from uberenv
#

# tbb from uberenv
#set(ASCENT_VTKM_USE_TBB OFF CACHE PATH "")
set(TBB_DIR "/usr/gapps/visit/strawman/uberenv_libs/spack/opt/spack/chaos_5_x86_64_ib/intel-14.0.3/tbb-4.4.3-al6fuqhyuhr6ju4daik3mfwk5j7gcyvw" CACHE PATH "")

# vtkm from uberenv
set(VTKM_DIR "/usr/workspace/wsb/larsen30/vtk-m/install" CACHE PATH "")

# hdf5 from uberenv
set(HDF5_DIR "/usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/hdf5-1.8.16-msbowehgkgvhlnl62fy6tb7bvefbr7h4" CACHE PATH "")

# silo from uberenv
set(SILO_DIR "/usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/silo-4.10.1-jnuhe4xm3vtwq4mevsobhahlriuqafrg" CACHE PATH "")

#SPHINX
set("SPHINX_EXECUTABLE" "/usr/gapps/conduit/thirdparty_libs/stable/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.9.3/python-2.7.11-eujx7frnxd5vpwolmye2fzq4tcylnbnv/bin/sphinx-build" CACHE PATH "")

##################################
# end uberenv host-config
##################################
