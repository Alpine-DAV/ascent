###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
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

###############################################################################
#
# Example that shows how to use an installed instance of BabelFlow PMT
#   mkdir build
#   cd build
#   cmake .. -DPMT_DIR=<PMT_DIR> -DBabelFlow_DIR=<BabelFlow_DIR>
#   make
#   mpirun -np 8 ascent_babelflow_example
#
##############################################################################


#cmake_minimum_required(VERSION 3.9)

#project(babelflow_pmt)

#set(ASCENT_DIR /home/xuanhuang/projects/ascent-update/ascent/build/install)
#set(BabelFlow_DIR /home/xuanhuang/package/babelflow-ascent/build/install/lib/cmake/BabelFlow)
#set(PMT_DIR /home/xuanhuang/projects/parallelmergetree/build/install/lib/cmake/PMT)


#
# Provide default for ASCENT_DIR that works for an ascent install
#
#set(ASCENT_DIR "" CACHE PATH "path to installed ascent")
#if (NOT ASCENT_DIR)
#    set(ASCENT_DIR "../../..")
#endif ()
#
# Check for valid Ascent install
#
#get_filename_component(ASCENT_DIR ${ASCENT_DIR} ABSOLUTE)
#if (NOT EXISTS ${ASCENT_DIR}/lib/cmake/AscentConfig.cmake)
#    MESSAGE(FATAL_ERROR "Could not find Ascent CMake include file (${ASCENT_DIR}/lib/cmake/AscentConfig.cmake)")
#endif ()

#
# Use CMake's find_package to import ascent's targets
#
#find_package(Ascent REQUIRED
#        NO_DEFAULT_PATH
#        PATHS ${ASCENT_DIR}/lib/cmake)

#set(PMT_DIR "" CACHE PATH "path to pmt installed dir")
#set(BabelFlow_DIR "" CACHE PATH "path to babelflow cmake dir")
#set(DIY_LIBRARIES "" CACHE PATH "path to libdiy.a")

#find_package(BabelFlow REQUIRED)
#find_package(PMT REQUIRED)
#find_package(MPI REQUIRED)

#include_directories(${ASCENT_DIR}/include/ascent ${MPI_C_INCLUDE_PATH} /home/xuanhuang/projects/parallelmergetree/diy/include)
#message("ASCENT_DIR ${ASCENT_DIR}")

# create our example
add_executable(ascent_babelflow_example ascent_babelflow_example.cpp)

# link to ascent
#target_link_directories(ascent_babelflow_example PUBLIC ${PMT_DIR}/lib)
target_link_libraries(ascent_babelflow_example ascent_mpi BabelFlow::babelflow ${MPI_CXX_LIBRARIES} PMT::pmt ${DIY_LIBRARIES})

