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

include(CMakeFindDependencyMacro)

###############################################################################
# Setup Conduit
###############################################################################
# If ZZZ_DIR not set, use known install path for Conduit and VTK-h
if(NOT CONDUIT_DIR)
    set(CONDUIT_DIR ${ASCENT_CONDUIT_DIR})
endif()

###############################################################################
# Check for CONDUIT_DIR
###############################################################################
if(NOT CONDUIT_DIR)
    MESSAGE(FATAL_ERROR "Could not find Conduit. Conduit requires explicit CONDUIT_DIR.")
endif()

if(NOT EXISTS ${CONDUIT_DIR}/lib/cmake/conduit/conduit.cmake)
    MESSAGE(FATAL_ERROR "Could not find Conduit CMake include file (${CONDUIT_DIR}/lib/cmake/conduit/conduit.cmake)")
endif()

###############################################################################
# Import Conduit's CMake targets
###############################################################################
find_dependency(Conduit REQUIRED
                NO_DEFAULT_PATH
                PATHS ${CONDUIT_DIR}/lib/cmake)

###############################################################################
# Setup VTK-h
###############################################################################
if(NOT VTKH_DIR)
    set(VTKH_DIR ${ASCENT_VTKH_DIR})
endif()
###############################################################################
# Setup VTKH if VTKM_DIR is set
###############################################################################
if(VTKH_DIR)
    if(NOT EXISTS ${VTKH_DIR}/lib/VTKhConfig.cmake)
        MESSAGE(FATAL_ERROR "Could not find VTKh CMake include file (${VTKH_DIR}/lib/VTKhConfig.cmake)")
    endif()

    ###############################################################################
    # Import vtk-h CMake targets
    ###############################################################################
    find_dependency(VTKh REQUIRED
                   NO_DEFAULT_PATH
                   PATHS ${VTKH_DIR}/lib/)
endif()

###############################################################################
# Setup Devil Ray
###############################################################################
if(NOT DRAY_DIR)
  set(DRAY_DIR ${ASCENT_DRAY_DIR})
endif()

if(DRAY_DIR)
  if(NOT EXISTS ${DRAY_DIR}/lib/cmake/DRayConfig.cmake)
    MESSAGE(FATAL_ERROR "Could not find Devil Ray CMake include file (${DRAY_DIR}/lib/cmake/DRayConfig.cmake)")
    endif()

    ###############################################################################
    # Import Devil Ray CMake targets
    ###############################################################################
    find_dependency(DRay REQUIRED
                   NO_DEFAULT_PATH
                   PATHS ${DRAY_DIR}/lib/cmake/)
endif()


###############################################################################
# Setup Umpire
###############################################################################
if(NOT UMPIRE_DIR)
  set(UMPIRE_DIR ${ASCENT_UMPIRE_DIR})
endif()

if(UMPIRE_DIR)
  if(NOT EXISTS ${UMPIRE_DIR}/share/umpire/cmake/)
    MESSAGE(FATAL_ERROR "Could not find Umpire CMake include file (${UMPIRE_DIR}/share/umpire/cmake/umpire-config.cmake)")
    endif()

    ###############################################################################
    # Import Devil Ray CMake targets
    ###############################################################################
    find_dependency(Umpire REQUIRED
                    NO_DEFAULT_PATH
                    PATHS ${UMPIRE_DIR}/share/umpire/cmake/)
endif()


###############################################################################
# MFEM (even serial) may require mpi, if so we need to find mpi
###############################################################################
if(ASCENT_MFEM_MPI_ENABLED AND NOT MPI_FOUND)
    find_package(MPI COMPONENTS CXX)
endif()
