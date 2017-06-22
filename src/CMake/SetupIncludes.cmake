###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Alpine. 
# 
# For details, see: http://software.llnl.gov/alpine/.
# 
# Please also read alpine/LICENSE
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


################################
#  Project Wide Includes
################################

# add lodepng include dir
include_directories(${PROJECT_SOURCE_DIR}/thirdparty_builtin/lodepng)

# add include dirs so units tests have access to the headers across
# libs and in unit tests

include_directories(${PROJECT_SOURCE_DIR}/alpine/)
include_directories(${PROJECT_BINARY_DIR}/alpine/)
include_directories(${PROJECT_SOURCE_DIR}/alpine/utils)
include_directories(${PROJECT_SOURCE_DIR}/alpine/pipelines)
include_directories(${PROJECT_SOURCE_DIR}/alpine/flow)
include_directories(${PROJECT_SOURCE_DIR}/alpine/flow/filters)
include_directories(${PROJECT_SOURCE_DIR}/alpine/pipelines/flow_filters)

include_directories(${CONDUIT_INCLUDE_DIRS})

if(VTKM_FOUND)
    # tbb
    if(TBB_FOUND)
        include_directories(${TBB_INCLUDE_DIRS})
    endif()
    # VTK-m
    include_directories(${VTKm_INCLUDE_DIRS})
endif()


if(MPI_FOUND)
    include_directories(${ICET_INCLUDE_DIRS})
    include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()


if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
endif()





