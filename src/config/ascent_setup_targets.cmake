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

set(ASCENT_INCLUDE_DIRS "${ASCENT_INSTALL_PREFIX}/include/ascent")

#
# Probe Ascent Features
#

if(ASCENT_SERIAL_ENABLED)
    # create convenience target that bundles all reg ascent deps (ascent::ascent)
    add_library(ascent::ascent INTERFACE IMPORTED)

    set_property(TARGET ascent::ascent
                 APPEND PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/")

    set_property(TARGET ascent::ascent
                 APPEND PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/ascent/")

    set_property(TARGET ascent::ascent
                 PROPERTY INTERFACE_LINK_LIBRARIES
                 ascent)
    # try to include conduit with new exports
    if(TARGET conduit::conduit)
        set_property(TARGET ascent::ascent
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     conduit::conduit)
    else()
        # if not, bottle conduit
        set_property(TARGET ascent::ascent
                     APPEND PROPERTY
                     INTERFACE_INCLUDE_DIRECTORIES ${CONDUIT_INCLUDE_DIRS})

        set_property(TARGET ascent::ascent
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     conduit conduit_relay conduit_blueprint)
    endif()

    if(ASCENT_VTKH_ENABLED)
        set_property(TARGET ascent::ascent
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     vtkh)
    endif()
endif()

# and if mpi enabled, a convenience target mpi case (ascent::cascent_mpi)
if(ASCENT_MPI_ENABLED)
    add_library(ascent::ascent_mpi INTERFACE IMPORTED)

    set_property(TARGET ascent::ascent_mpi
                 APPEND PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/")

    set_property(TARGET ascent::ascent_mpi
                 APPEND PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/ascent/")

    set_property(TARGET ascent::ascent_mpi
                 PROPERTY INTERFACE_LINK_LIBRARIES
                 ascent_mpi)
    # try to include conduit with new exports
    if(TARGET conduit::conduit)
        set_property(TARGET ascent::ascent_mpi
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     conduit::conduit conduit::conduit_mpi)
    else()
        # if not, bottle conduit
        set_property(TARGET ascent::ascent_mpi
                     APPEND PROPERTY
                     INTERFACE_INCLUDE_DIRECTORIES ${CONDUIT_INCLUDE_DIRS})

        set_property(TARGET ascent::ascent_mpi
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     conduit conduit_relay conduit_relay_mpi conduit_blueprint)
    endif()

    if(ASCENT_VTKH_ENABLED)
        set_property(TARGET ascent::ascent_mpi
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     vtkh_mpi)
    endif()


endif()

if(NOT Ascent_FIND_QUIETLY)

    message(STATUS "ASCENT_VERSION             = ${ASCENT_VERSION}")
    message(STATUS "ASCENT_INSTALL_PREFIX      = ${ASCENT_INSTALL_PREFIX}")
    message(STATUS "ASCENT_INCLUDE_DIRS        = ${ASCENT_INCLUDE_DIRS}")
    message(STATUS "ASCENT_SERIAL_ENABLED      = ${ASCENT_SERIAL_ENABLED}")
    message(STATUS "ASCENT_MPI_ENABLED         = ${ASCENT_MPI_ENABLED}")
    message(STATUS "ASCENT_FORTRAN_ENABLED     = ${ASCENT_FORTRAN_ENABLED}")
    message(STATUS "ASCENT_VTKH_ENABLED        = ${ASCENT_VTKH_ENABLED}")
    message(STATUS "ASCENT_PYTHON_ENABLED      = ${ASCENT_PYTHON_ENABLED}")
    message(STATUS "ASCENT_PYTHON_EXECUTABLE   = ${ASCENT_PYTHON_EXECUTABLE}")
    message(STATUS "ASCENT_DRAY_ENABLED        = ${ASCENT_DRAY_ENABLED}")
    message(STATUS "ASCENT_MFEM_ENABLED        = ${ASCENT_MFEM_ENABLED}")
    message(STATUS "ASCENT_MFEM_MPI_ENABLED    = ${ASCENT_MFEM_MPI_ENABLED}")


    set(_print_targets "")
    if(ASCENT_SERIAL_ENABLED)
        set(_print_targets "ascent::ascent ")
    endif()

    if(ASCENT_MPI_ENABLED)
        set(_print_targets "${_print_targets}ascent::ascent_mpi")
    endif()

    message(STATUS "Ascent imported targets: ${_print_targets}")
    unset(_print_targets)

endif()


