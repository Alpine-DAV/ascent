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

##################################
# empty  host-config
##################################
# insert compiler name here
##################################

#######
# using [insert compiler name here] compiler spec
#######

# c compiler
set(CMAKE_C_COMPILER "/path/to/c_compiler" CACHE PATH "")

# cpp compiler
set(CMAKE_CXX_COMPILER "/path/to/cxx_compiler" CACHE PATH "")

# fortran compiler (need for cloverleaf)
set(CMAKE_Fortran_COMPILER  "/path/to/fortran_compiler" CACHE PATH "")

# OPENMP (optional: for proxy apps and EAVL)
set(ENABLE_OPENMP ON CACHE PATH "")

# MPI Support
set(ENABLE_MPI  ON CACHE PATH "")

set(MPI_C_COMPILER  "/path/to/mpi_c_compiler" CACHE PATH "")

set(MPI_CXX_COMPILER "/path/to/mpi_cxx_compiler" CACHE PATH "")

set(MPI_Fortran_COMPILER "/path/to/fortran90_compiler" CACHE PATH "")

set(MPIEXEC /usr/bin/srun CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG -n CACHE PATH "")



# CUDA support
#set(ENABLE_CUDA ON CACHE PATH "")
#set(CUDA_BIN_DIR /path/to/cudatoolkit-7.0/bin CACHE PATH "")

# NO CUDA Support
set(ENABLE_CUDA OFF CACHE PATH "")


# conduit 
set(CONDUIT_DIR "/path/to/conduit_install/" CACHE PATH "")

# icet 
set(ICET_DIR "/path/to/icet_install/" CACHE PATH "")

#
# vtkm
#

# boost-headers 
set(BOOST_DIR "/path/to/boost/" CACHE PATH "")

# tbb
#set(STRAWMAN_VTKM_USE_TBB OFF CACHE PATH "")
set(TBB_DIR "/path/to/tbb_install" CACHE PATH "")

# vtkm
set(VTKM_DIR "/path/to/vtkm_install" CACHE PATH "")

#
# eavl support (optional)
#

# osmesa 
set(OSMESA_DIR "/path/to/osmesa_install" CACHE PATH "")

# eavl
set(EAVL_DIR "/path/to/eavl_install" CACHE PATH "")

# 
# HDF5 support (optional)
#
# hdf5v
set(HDF5_DIR "/path/to/hdf5_install" CACHE PATH "")

#SPHINX documentation building
set("SPHINX_EXECUTABLE" "/path/to/sphinx-build" CACHE PATH "")

##################################
# end boilerplate host-config
##################################
