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
# About Util CMake Build for Ascent
#
###############################################################################

set(ABOUT_SOURCES
    about.cpp)

set(ABOUT_DEPS ascent)

if(OPENMP_FOUND)
   list(APPEND deps openmp)
endif()

if (ENABLE_SERIAL)
    blt_add_executable(
        NAME        ascent_about
        SOURCES     ${ABOUT_SOURCES}
        DEPENDS_ON  ${ABOUT_DEPS}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    
    # install target for replay serial
    install(TARGETS ascent_about
            EXPORT  ascent
            LIBRARY DESTINATION utilities/ascent/
            ARCHIVE DESTINATION utilities/ascent/
            RUNTIME DESTINATION utilities/ascent/
    )
endif()

if(MPI_FOUND)

    set(ABOUT_MPI_DEPS ascent_mpi mpi)
    if(OPENMP_FOUND)
           list(APPEND about_mpi_deps openmp)
    endif()

    blt_add_executable(
        NAME        ascent_about_mpi
        SOURCES     ${ABOUT_SOURCES}
        DEPENDS_ON  ${ABOUT_MPI_DEPS}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    blt_add_target_compile_flags(TO ascent_about_mpi FLAGS "-DABOUT_MPI=1")

    # install target for replay mpi
    install(TARGETS ascent_about_mpi
            EXPORT  ascent
            LIBRARY DESTINATION utilities/ascent/
            ARCHIVE DESTINATION utilities/ascent/
            RUNTIME DESTINATION utilities/ascent/
    )

endif()






