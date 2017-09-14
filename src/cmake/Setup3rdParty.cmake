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

################################
# Ascent 3rd Party Dependencies
################################

###############################################################################
# gtest, fruit, mpi,cuda, openmp, sphinx and doxygen are handled by blt
###############################################################################

################################
# Setup Python if requested
################################
if(ENABLE_PYTHON)
    include(cmake/thirdparty/SetupPython.cmake)
    message(STATUS "Using Python Include: ${PYTHON_INCLUDE_DIRS}")
    include_directories(${PYTHON_INCLUDE_DIRS})
    # if we don't find python, throw a fatal error
    if(NOT PYTHON_FOUND)
        message(FATAL_ERROR "ENABLE_PYTHON is true, but Python wasn't found.")
    endif()
endif()

################################
# Conduit
################################
include(cmake/thirdparty/SetupConduit.cmake)


################################################################
################################################################
#
# 3rd Party Libs that underpin Ascent's Pipelines
#
################################################################
################################################################


################################
# VTKm and supporting libs
################################
if(VTKM_DIR)
    # explicitly setting this avoids a bug with VTKm's cuda
    # arch detection logic
    set(VTKm_CUDA_Architecture "kepler" CACHE PATH "" FORCE)

    ################################
    # TBB (for VTK-M)
    ################################
    if(TBB_DIR) # optional 
        include(cmake/thirdparty/SetupTBB.cmake)
    endif()

    ################################
    # VTKm
    ################################
    include(cmake/thirdparty/SetupVTKm.cmake)
endif()


################################
# Setup HDF5
################################
if(HDF5_DIR)
    include(cmake/thirdparty/SetupHDF5.cmake)
endif()


################################
# Optional Features
################################

################################
# IceT
################################
if(ENABLE_MPI AND VTKM_FOUND)
    include(cmake/thirdparty/SetupIceT.cmake)
endif()


