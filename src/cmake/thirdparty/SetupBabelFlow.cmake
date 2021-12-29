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
# Setup BabelFlow and ParallelMergeTree
###############################################################################


# allow prev case style "BabelFlow_DIR"
if(NOT BABELFLOW_DIR)
    if(BabelFlow_DIR)
        set(BABELFLOW_DIR ${BabelFlow_DIR})
    endif()
endif()

if(NOT BABELFLOW_DIR)
    MESSAGE(FATAL_ERROR "BabelFlow support needs explicit BABELFLOW_DIR")
endif()


MESSAGE(STATUS "Looking for BabelFlow using BABELFLOW_DIR = ${BABELFLOW_DIR}")

set(BABELFLOW_DIR_ORIG ${BABELFLOW_DIR})

find_package(BabelFlow REQUIRED
             NO_DEFAULT_PATH
             PATHS ${BABELFLOW_DIR}/lib/cmake/)

message(STATUS "FOUND BabelFlow at ${BABELFLOW_DIR}")

set(BABELFLOW_FOUND TRUE)

blt_register_library( NAME babelflow
                      INCLUDES ${BabelFlow_INCLUDE_DIRS}
                      LIBRARIES  babelflow babelflow_mpi)

## Find also ParallelMergeTree analysis algorithm to build (based on BabelFlow)
if(NOT PMT_DIR)
    MESSAGE(FATAL_ERROR "ParallelMergeTree support needs explicit PMT_DIR")
endif()

MESSAGE(STATUS "Looking for ParallelMergeTree using PMT_DIR = ${PMT_DIR}")

set(PMT_DIR_ORIG ${PMT_DIR})

find_package(PMT REQUIRED
             NO_DEFAULT_PATH
             PATHS ${PMT_DIR}/lib/cmake)

message(STATUS "FOUND PMT at ${PMT_DIR}")

blt_register_library( NAME pmt
                      INCLUDES ${PMT_INCLUDE_DIRS}
                      LIBRARIES  pmt)
