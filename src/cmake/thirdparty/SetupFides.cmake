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
# Setup Fides
#
###############################################################################

if(NOT FIDES_DIR)
    MESSAGE(FATAL_ERROR "Fides support needs explicit FIDES_DIR")
endif()

if(NOT VTKM_DIR)
    MESSAGE(FATAL_ERROR "Fides support needs VTK-m (VTKM_DIR not set)")
endif()

if(NOT ADIOS2_DIR)
    MESSAGE(FATAL_ERROR "Fides support needs ADIOS2 (ADIOS2_DIR not set)")
endif()

MESSAGE(STATUS "Looking for FIDES using FIDES_DIR = ${FIDES_DIR}")

set(FIDES_DIR_ORIG ${FIDES_DIR})

#The Fides cmake is not setting these for some reason.
#So, we set them explicitly for now.
#set(Fides_DIR ${FIDES_DIR})

set(FIDES_INCLUDE_DIR ${FIDES_DIR}/include/)
set(FIDES_LIB_DIR ${FIDES_DIR}/lib)
set(FIDES_LIBRARIES fides)

find_package(Fides REQUIRED
             NO_DEFAULT_PATH
             PATHS ${FIDES_DIR}/lib/cmake/fides)


message(STATUS "Found Fides at ${FIDES_DIR}")
set(FIDES_FOUND TRUE)

blt_register_library(NAME fides
                     INCLUDES ${FIDES_INCLUDE_DIR}
                     LIBRARIES ${FIDES_LIB_DIRS} ${FIDES_LIBRARIES} )
